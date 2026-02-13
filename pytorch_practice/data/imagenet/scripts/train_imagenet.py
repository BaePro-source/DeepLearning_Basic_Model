import os
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent            # .../data/imagenet/scripts
imagenet_root = script_dir.parent                       # .../data/imagenet
data_dir = imagenet_root / "data"                       # .../data/imagenet/data
sys.path.append(str(imagenet_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

from models.vgg16 import VGG16ImageNet


def train_one_epoch(model, loader, criterion, optimizer, device, scaler) :
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # 🔥 AMP 핵심 구간
        with torch.cuda.amp.autocast():
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

        # 로그 (30분 무출력 방지)
        if step % 10 == 0:
            print(f"[train] step={step} loss={loss.item():.4f}", flush=True)

    return total_loss / max(total, 1), correct / max(total, 1)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            logits = model(x)

        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return correct / max(total, 1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # -------------------------
    # Transform
    # -------------------------
    train_tf = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    val_tf = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_dir = data_dir / "split" / "train"
    val_dir = data_dir / "split" / "val"

    print("[path]")
    print("train_dir:", train_dir)
    print("val_dir  :", val_dir)
    print("train exists:", train_dir.is_dir())
    print("val exists  :", val_dir.is_dir())
    print()

    if not train_dir.is_dir():
        raise RuntimeError(f"TRAIN_DIR not found: {train_dir}")
    if not val_dir.is_dir():
        raise RuntimeError(f"VAL_DIR not found: {val_dir}")

    # -------------------------
    # Dataset / Loader (ImageFolder)
    # -------------------------
    train_ds = ImageFolder(str(train_dir), transform=train_tf)
    val_ds = ImageFolder(str(val_dir), transform=val_tf)

    # 클래스 수는 폴더 기준으로 자동
    num_classes = len(train_ds.classes)
    if num_classes != 1000:
        print(f"[warn] num_classes={num_classes} (expected 1000). 폴더 구조 확인 필요")

    print("[dataset]")
    print("num_classes:", num_classes)
    print("train samples:", len(train_ds))
    print("val samples  :", len(val_ds))
    print()

    train_loader = DataLoader(
        train_ds,
        batch_size=32,
        shuffle=True,
        num_workers=2,   
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # -------------------------
    # Model
    # -------------------------
    model = VGG16ImageNet(num_classes=num_classes).to(device)

    # forward check
    x, y = next(iter(train_loader))
    with torch.no_grad():
        out = model(x.to(device))
    print("[forward check]", tuple(x.shape), "->", tuple(out.shape))
    print()

    # -------------------------
    # Train
    # -------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scaler = torch.cuda.amp.GradScaler()

    start_epoch = 10
    epochs = 20
    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_acc = evaluate(model, val_loader, device)
        print(f"[Epoch {epoch}] loss={train_loss:.4f} train_acc={train_acc*100:.2f}% val_acc={val_acc*100:.2f}%")
        ckpt_path = imagenet_root / "checkpoints" / f"vgg16_epoch{epoch}.pth"
        torch.save(model.state_dict(), str(ckpt_path))
        print(f"[checkpoint saved] {ckpt_path}")


if __name__ == "__main__":
    main()
