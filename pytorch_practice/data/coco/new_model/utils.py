"""
Training and Evaluation Utilities
"""
import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from PIL import Image


@torch.no_grad()
def compute_class_weights(loader, num_classes, device):
    """Compute class weights based on frequency"""
    counts = torch.zeros(num_classes, dtype=torch.long)
    
    for _, y in loader:
        y = y.view(-1)
        for c in y.unique():
            if c < num_classes:
                counts[c] += (y == c).sum()
    
    freq = counts.float() / counts.sum().clamp(min=1)
    weights = 1.0 / torch.log(1.02 + freq)
    return weights.to(device)


@torch.no_grad()
def update_confmat(confmat, pred, target, num_classes):
    """Update confusion matrix"""
    pred = pred.view(-1).to(torch.int64)
    target = target.view(-1).to(torch.int64)

    valid = (target >= 0) & (target < num_classes) & (pred >= 0) & (pred < num_classes)
    pred = pred[valid]
    target = target[valid]

    idx = target * num_classes + pred
    confmat += torch.bincount(idx, minlength=num_classes * num_classes).cpu().view(num_classes, num_classes)


def miou_from_confmat(confmat: torch.Tensor):
    """Calculate mIoU from confusion matrix"""
    confmat = confmat.to(torch.float64)

    tp = torch.diag(confmat)
    fp = confmat.sum(0) - tp
    fn = confmat.sum(1) - tp
    denom = tp + fp + fn

    iou = tp / torch.clamp(denom, min=1.0)
    miou_all = iou.mean().item()

    present = denom > 0
    miou_present = iou[present].mean().item() if present.any() else 0.0

    return miou_all, miou_present, iou


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int, print_topk: int = 0):
    """Evaluate model on validation set"""
    model.eval()
    confmat = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        pred = logits.argmax(dim=1)

        update_confmat(confmat, pred.cpu(), y.cpu(), num_classes)

    # Calculate metrics
    correct = torch.diag(confmat).sum().item()
    total = confmat.sum().item()
    acc = correct / max(total, 1)

    miou_all, miou_present, iou = miou_from_confmat(confmat)

    # Print top/bottom classes by IoU
    if print_topk > 0:
        iou_np = iou.cpu()
        denom = (torch.diag(confmat) + (confmat.sum(0)-torch.diag(confmat)) + 
                 (confmat.sum(1)-torch.diag(confmat))).cpu()
        present = denom > 0

        idx_present = torch.where(present)[0]
        iou_present_vec = iou_np[idx_present]

        k = min(print_topk, len(idx_present))
        if k > 0:
            topv, topi = torch.topk(iou_present_vec, k=k, largest=True)
            top_cls = idx_present[topi]
            lowv, lowi = torch.topk(iou_present_vec, k=k, largest=False)
            low_cls = idx_present[lowi]

            print(f"[IoU TOP {k}] " + ", ".join([f"{int(c)}:{float(v)*100:.2f}%" 
                                                   for c, v in zip(top_cls, topv)]))
            print(f"[IoU LOW {k}] " + ", ".join([f"{int(c)}:{float(v)*100:.2f}%" 
                                                   for c, v in zip(low_cls, lowv)]))

    return acc, miou_all, miou_present


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def denormalize(img_chw: torch.Tensor) -> np.ndarray:
    """Denormalize image tensor to uint8 RGB"""
    img = img_chw.detach().cpu()
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    img = (img * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def make_palette(num_classes: int, seed: int = 123) -> np.ndarray:
    """Generate color palette for visualization"""
    rng = np.random.default_rng(seed)
    pal = rng.integers(0, 256, size=(num_classes, 3), dtype=np.uint8)
    pal[0] = np.array([0, 0, 0], dtype=np.uint8)  # Background = black
    return pal


@torch.no_grad()
def save_predictions(model, loader, device, num_classes: int, 
                     out_dir="vis", max_images=4, alpha=0.45):
    """Save visualization of predictions"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    palette = make_palette(num_classes)
    model.eval()

    saved = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        pred = logits.argmax(dim=1)

        x_cpu = x.cpu()
        y_cpu = y.cpu().numpy().astype(np.int32)
        p_cpu = pred.cpu().numpy().astype(np.int32)

        for i in range(x_cpu.size(0)):
            img = denormalize(x_cpu[i])
            gt = y_cpu[i]
            pr = p_cpu[i]

            gt_col = palette[np.clip(gt, 0, num_classes-1)]
            pr_col = palette[np.clip(pr, 0, num_classes-1)]

            overlay_gt = (img * (1-alpha) + gt_col * alpha).astype(np.uint8)
            overlay_pred = (img * (1-alpha) + pr_col * alpha).astype(np.uint8)

            prefix = f"{saved:03d}"
            Image.fromarray(img).save(out_dir / f"{prefix}_img.png")
            Image.fromarray(gt_col).save(out_dir / f"{prefix}_gt.png")
            Image.fromarray(pr_col).save(out_dir / f"{prefix}_pred.png")
            Image.fromarray(overlay_gt).save(out_dir / f"{prefix}_overlay_gt.png")
            Image.fromarray(overlay_pred).save(out_dir / f"{prefix}_overlay_pred.png")

            saved += 1
            if saved >= max_images:
                print(f"[VIS] Saved {saved} images to {out_dir.resolve()}")
                return

    print(f"[VIS] Saved {saved} images to {out_dir.resolve()}")