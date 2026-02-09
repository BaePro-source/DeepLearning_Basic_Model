import os
import re
import glob
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

# ✅ 여기만 본인 모델로 맞추세요 (커스텀 VGG라면 이 줄 사용)
from models.vgg16 import VGG16ImageNet

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def parse_epoch(p: str) -> int:
    m = re.search(r"epoch(\d+)", os.path.basename(p))
    return int(m.group(1)) if m else -1

@torch.no_grad()
def eval_topk(model, loader, device, k=5):
    model.eval()
    top1 = 0
    topk = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)

        # top-1
        pred1 = logits.argmax(dim=1)
        top1 += (pred1 == y).sum().item()

        # top-k
        predk = logits.topk(k, dim=1).indices  # (B, k)
        topk += predk.eq(y.view(-1, 1)).any(dim=1).sum().item()

        total += y.size(0)

    return top1 / max(total, 1), topk / max(total, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_dir", type=str, required=True, help="e.g. data/split/val")
    ap.add_argument("--ckpt_glob", type=str, required=True, help="e.g. checkpoints/vgg16_epoch*.pth")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tf = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_ds = ImageFolder(args.val_dir, transform=tf)
    num_classes = len(val_ds.classes)

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    ckpts = sorted(glob.glob(args.ckpt_glob), key=parse_epoch)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints matched: {args.ckpt_glob}")

    rows = []
    for ckpt_path in ckpts:
        epoch = parse_epoch(ckpt_path)

        model = VGG16ImageNet(num_classes=num_classes).to(device)
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state)

        top1, top5 = eval_topk(model, val_loader, device, k=5)
        rows.append((epoch, top1 * 100, top5 * 100))
        print(f"[eval] epoch={epoch} top1={top1*100:.2f}% top5={top5*100:.2f}%  ({ckpt_path})")

    # 표 출력 (에폭 순 정렬)
    rows.sort(key=lambda x: x[0])

    print("\n=== VGG16 Accuracy Table ===")
    print("| Epoch | Top-1 Acc (%) | Top-5 Acc (%) |")
    print("|------:|--------------:|--------------:|")
    for e, t1, t5 in rows:
        print(f"| {e} | {t1:>10.2f} | {t5:>10.2f} |")

if __name__ == "__main__":
    main()
