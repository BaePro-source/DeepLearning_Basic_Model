import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt

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
def evaluate(model, loader, criterion, device):
    model.eval()
    train_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            logits = model(x)
            loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0) 
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def plot_training_curves(history, model_name):
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    plt.figure()
    plt.plot(epochs, history["train_loss"])
    plt.plot(epochs, history["val_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} - Loss Curve")
    plt.legend(["Train", "Validation"])
    plt.grid(True)
    plt.savefig(f"{model_name}_loss_curve.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_acc"])
    plt.plot(epochs, history["val_acc"])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} - Accuracy Curve")
    plt.legend(["Train", "Validation"])
    plt.grid(True)
    plt.savefig(f"{model_name}_acc_curve.png", dpi=300)
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    history = {
        "train_loss" : [],
        "val_loss" : [],
        "train_acc" : [],
        "val_acc" : []
    }

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

    start_epoch = 1
    epochs = 20
    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"[Epoch {epoch}] "
            f"loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"train_acc={train_acc*100:.2f}% "
            f"val_acc={val_acc*100:.2f}%"
        )

    plot_training_curves(history, "VGG16", start_epoch)


if __name__ == "__main__":
    main()
