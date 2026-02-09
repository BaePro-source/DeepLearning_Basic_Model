import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import numpy as np

from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from torch.utils.data import Subset
from PIL import Image

import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torchvision.models as models


class ASPP(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 256, rates=(6, 12, 18)):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.branches_atrous = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
            for r in rates
        ])
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        total = out_ch * (2 + len(rates))
        self.project = nn.Sequential(
            nn.Conv2d(total, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        h, w = x.shape[2:]
        feats = [self.branch1(x)]
        feats += [b(x) for b in self.branches_atrous]
        pool = self.image_pool(x)
        pool = F.interpolate(pool, size=(h, w), mode="bilinear", align_corners=False)
        feats.append(pool)
        x = torch.cat(feats, dim=1)
        return self.project(x)


class DeepLabV3Plus(nn.Module):
    """
    - backbone: resnet50
    - ASPP
    - low-level skip (layer1)
    - decoder
    """
    def __init__(self, num_classes: int, pretrained: bool = True, output_stride: int = 16):
        super().__init__()
        if output_stride == 16:
            replace_stride = [False, False, True]
            aspp_rates = (6, 12, 18)
        elif output_stride == 8:
            replace_stride = [False, True, True]
            aspp_rates = (12, 24, 36)
        else:
            raise ValueError("output_stride는 8 또는 16")

        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = models.resnet50(weights=weights, replace_stride_with_dilation=replace_stride)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1  # low-level (256ch)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4  # high-level (2048ch)

        self.aspp = ASPP(2048, 256, rates=aspp_rates)

        # low-level projection
        self.low_proj = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        input_hw = x.shape[2:]

        x = self.stem(x)
        low = self.layer1(x)       # [B,256,H/4,W/4]
        x = self.layer2(low)
        x = self.layer3(x)
        x = self.layer4(x)         # [B,2048, ... ]

        x = self.aspp(x)
        x = F.interpolate(x, size=low.shape[2:], mode="bilinear", align_corners=False)

        low = self.low_proj(low)
        x = torch.cat([x, low], dim=1)
        x = self.decoder(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=input_hw, mode="bilinear", align_corners=False)
        return x


# -------------------------
# 2) COCO -> Semantic Mask Dataset
# -------------------------
class COCOSemSeg(Dataset):
    """
    COCO instance annotations -> semantic mask
    - mask: [H,W] (long), background=0, classes=1..K
    - ignore_index: 255 (옵션)
    """
    def __init__(self, img_dir, ann_json, resize=512, crop=512, train=True, seed=42, ignore_index=255):
        self.img_dir = Path(img_dir)
        self.coco = COCO(str(ann_json))
        self.img_ids = sorted(self.coco.getImgIds())
        self.train = train
        self.resize = resize
        self.crop = crop
        self.ignore_index = ignore_index

        # COCO category id -> contiguous id (1..K)
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.cat_ids = sorted([c["id"] for c in cats])
        self.cat2contig = {cat_id: (i + 1) for i, cat_id in enumerate(self.cat_ids)}  # 0은 background

        random.seed(seed)

    def __len__(self):
        return len(self.img_ids)

    def _anns_to_mask(self, img_info):
        h, w = img_info["height"], img_info["width"]
        mask = np.zeros((h, w), dtype=np.uint8)  # background=0

        ann_ids = self.coco.getAnnIds(imgIds=img_info["id"], iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)

        # 단순 규칙: 뒤에 나온 객체가 덮어씀(겹치면 overwrite)
        for a in anns:
            cat_id = a["category_id"]
            contig = self.cat2contig.get(cat_id, 0)
            if contig == 0:
                continue

            seg = a.get("segmentation", None)
            if seg is None:
                continue

            if isinstance(seg, list):
                rles = maskUtils.frPyObjects(seg, h, w)
                rle = maskUtils.merge(rles)
            elif isinstance(seg, dict) and "counts" in seg:
                rle = seg
            else:
                continue

            m = maskUtils.decode(rle)  # (h,w) {0,1}
            mask[m == 1] = contig

        return mask  # uint8

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        img_path = self.img_dir / img_info["file_name"]

        image = Image.open(img_path).convert("RGB")
        mask_np = self._anns_to_mask(img_info)
        mask = Image.fromarray(mask_np, mode="L")  # 0..K

        # ---- 동일 변환(이미지/마스크 같이) ----
        # 1) resize
        image = TF.resize(image, [self.resize, self.resize], interpolation=Image.BILINEAR)
        mask = TF.resize(mask, [self.resize, self.resize], interpolation=Image.NEAREST)

        if self.train:
            # 2) random crop
            i, j, h, w = T.RandomCrop.get_params(image, output_size=(self.crop, self.crop))
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

            # 3) random flip
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
        else:
            # val은 center crop(간단)
            image = TF.center_crop(image, [self.crop, self.crop])
            mask = TF.center_crop(mask, [self.crop, self.crop])

        # ---- tensor 변환 ----
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        mask = torch.from_numpy(np.array(mask, dtype=np.uint8)).long()  # [H,W]

        return image, mask


# -------------------------
# 3) Train / Eval
# -------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)                 # [B,C,H,W]
        pred = logits.argmax(dim=1)       # [B,H,W]
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def train_one_epoch(model, loader, opt, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)

def denorm(img_chw: torch.Tensor) -> np.ndarray:
    """normalize된 Tensor[3,H,W] -> uint8 RGB(H,W,3)"""
    img = img_chw.detach().cpu()
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    img = (img * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

def make_palette(num_classes: int, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pal = rng.integers(0, 256, size=(num_classes, 3), dtype=np.uint8)
    pal[0] = np.array([0, 0, 0], dtype=np.uint8)  # bg=black
    return pal

@torch.no_grad()
def save_predictions(model, loader, device, num_classes: int, out_dir="vis", max_images=4, alpha=0.45):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    palette = make_palette(num_classes)
    model.eval()

    saved = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        out = model(x)
        logits = out["out"] if isinstance(out, dict) else out
        pred = logits.argmax(dim=1)  # [B,H,W]

        x_cpu = x.cpu()
        y_cpu = y.cpu().numpy().astype(np.int32)
        p_cpu = pred.cpu().numpy().astype(np.int32)

        for i in range(x_cpu.size(0)):
            img = denorm(x_cpu[i])                 # (H,W,3)
            gt  = y_cpu[i]
            pr  = p_cpu[i]

            gt_col = palette[np.clip(gt, 0, num_classes-1)]
            pr_col = palette[np.clip(pr, 0, num_classes-1)]

            overlay_gt   = (img * (1-alpha) + gt_col * alpha).astype(np.uint8)
            overlay_pred = (img * (1-alpha) + pr_col * alpha).astype(np.uint8)

            prefix = f"{saved:03d}"
            Image.fromarray(img).save(out_dir / f"{prefix}_img.png")
            Image.fromarray(gt_col).save(out_dir / f"{prefix}_gt.png")
            Image.fromarray(pr_col).save(out_dir / f"{prefix}_pred.png")
            Image.fromarray(overlay_gt).save(out_dir / f"{prefix}_overlay_gt.png")
            Image.fromarray(overlay_pred).save(out_dir / f"{prefix}_overlay_pred.png")

            saved += 1
            if saved >= max_images:
                print(f"[VIS] saved {saved} images to {out_dir.resolve()}")
                return

    print(f"[VIS] saved {saved} images to {out_dir.resolve()}")


def main():
    # ===== 여기만 본인 COCO 경로로 바꾸세요 =====
    coco_root = Path(__file__).resolve().parent
    train_img = coco_root / "images" / "test" / "train2017"
    val_img = coco_root / "images" / "val"/ "val2017"
    train_ann = coco_root / "annotations" / "stuff_train2017.json"
    val_ann = coco_root / "annotations" / "stuff_val2017.json"

    # 하이퍼파라미터(일단 돌아가는 값)
    resize = 520
    crop = 512
    batch_size = 2
    num_workers = 2
    lr = 1e-4
    epochs = 50
    output_stride = 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    
    base_train_ds = COCOSemSeg(train_img, train_ann, resize=resize, crop=crop, train=True)
    base_val_ds   = COCOSemSeg(val_img, val_ann, resize=resize, crop=crop, train=False)

    N_TRAIN = 64   # 예: 64장만
    N_VAL   = 128   # 예: 16장만

    # train_ds = COCOSemSeg(train_img, train_ann, resize=resize, crop=crop, train=True)
    # val_ds = COCOSemSeg(val_img, val_ann, resize=resize, crop=crop, train=False)

    train_ds = Subset(base_train_ds, list(range(min(N_TRAIN, len(base_train_ds)))))
    val_ds   = Subset(base_val_ds,   list(range(min(N_VAL,   len(base_val_ds)))))

    num_classes = 1 + len(base_train_ds.cat_ids)  # background 포함(0)
    print("num_classes (bg 포함):", num_classes)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last = True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=0, pin_memory=True)

    model = DeepLabV3Plus(num_classes=num_classes, pretrained=True, output_stride=output_stride).to(device)

    criterion = nn.CrossEntropyLoss()  # softmax 포함이라 logits 그대로 넣음
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    save_predictions(model, val_loader, device, num_classes, out_dir="vis/before", max_images=4)

    # 빠른 sanity check (shape)
    x0, y0 = next(iter(train_loader))
    print("x:", x0.shape, "y:", y0.shape)
    with torch.no_grad():
        out0 = model(x0.to(device))
    print("out:", out0.shape, "pred:", out0.argmax(1).shape)

    best_val_acc = 0.0

    for ep in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, opt, criterion, device)

        train_acc = evaluate(model, train_loader, device)
        val_acc   = evaluate(model, val_loader, device)

        print(f"[Epoch {ep}] loss={loss:.4f}")
        print(f"train-acc={train_acc*100:.2f}%  val-acc={val_acc*100:.2f}%")


        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best.pth")

    # 저장
    ckpt = Path("deeplabv3plus_coco_semseg.pth")
    torch.save(model.state_dict(), ckpt)
    print("saved:", ckpt)

    # 학습 끝난 뒤
    model.load_state_dict(torch.load("best.pth", map_location=device))
    model.eval()

    save_predictions(
        model,
        val_loader,
        device,
        num_classes,
        out_dir="vis/best",
        max_images=4
)



if __name__ == "__main__":
    main()
