# data/imagenet/scripts/check_imagenet_loader.py
# ImageNet split/train, split/val (wnid 폴더 구조) 로더 체크 스크립트
# - CWD(실행 위치) 상관없이 동작
# - torchvision.datasets.ImageFolder 사용
# - (옵션) split/val에서 45k/5k 서브분할 가능

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
from torchvision.datasets import ImageFolder


class TinyDummy(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(8, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def make_transforms(img_size: int = 224):
    # ImageNet val transform(기본)
    tf = T.Compose([
        T.Resize(256),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])
    return tf


def resolve_dirs():
    """
    현재 프로젝트 구조(사용자 기준):
    PYTORCH_PRACTICE/
      data/
        imagenet/
          data/
            split/
              train/
              val/
          scripts/
            check_imagenet_loader.py
    """
    SCRIPT_DIR = Path(__file__).resolve().parent          # .../data/imagenet/scripts
    IMAGENET_ROOT = SCRIPT_DIR.parent                     # .../data/imagenet
    DATA_DIR = IMAGENET_ROOT / "data"                     # .../data/imagenet/data
    SPLIT_DIR = DATA_DIR / "split"
    TRAIN_DIR = SPLIT_DIR / "train"
    VAL_DIR = SPLIT_DIR / "val"
    return IMAGENET_ROOT, DATA_DIR, TRAIN_DIR, VAL_DIR


def print_dir_summary(p: Path, name: str, max_show: int = 8):
    print(f"[{name}] {p}")
    print("exists:", p.exists(), "is_dir:", p.is_dir())
    if p.is_dir():
        subdirs = sorted([d.name for d in p.iterdir() if d.is_dir()])
        print("num_class_dirs:", len(subdirs))
        if subdirs:
            print("first few class dirs:", subdirs[:max_show])
    print()


def build_loader(ds, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)  # Windows 디버깅은 0 추천
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--subset_val", action="store_true",
                        help="split/val(50000)을 45000/5000으로 추가 분할해 체크")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_size", type=int, default=45000)
    parser.add_argument("--val_size", type=int, default=5000)
    args = parser.parse_args()

    IMAGENET_ROOT, DATA_DIR, TRAIN_DIR, VAL_DIR = resolve_dirs()

    print("=== Path Check ===")
    print("IMAGENET_ROOT:", IMAGENET_ROOT)
    print("DATA_DIR     :", DATA_DIR)
    print_dir_summary(TRAIN_DIR, "TRAIN_DIR")
    print_dir_summary(VAL_DIR, "VAL_DIR")

    if not TRAIN_DIR.is_dir():
        raise RuntimeError(f"TRAIN_DIR not found: {TRAIN_DIR}")
    if not VAL_DIR.is_dir():
        raise RuntimeError(f"VAL_DIR not found: {VAL_DIR}")

    tf = make_transforms(args.img_size)

    # ✅ ImageFolder: 클래스별 폴더 구조가 있어야 함
    train_ds = ImageFolder(str(TRAIN_DIR), transform=tf)
    val_ds = ImageFolder(str(VAL_DIR), transform=tf)

    print("=== Dataset Check (ImageFolder) ===")
    print("train classes:", len(train_ds.classes), "samples:", len(train_ds.samples))
    print("val   classes:", len(val_ds.classes), "samples:", len(val_ds.samples))
    print("train class_to_idx sample:", list(train_ds.class_to_idx.items())[:3])
    print("val   class_to_idx sample:", list(val_ds.class_to_idx.items())[:3])
    print()

    # 배치 체크
    train_loader = build_loader(train_ds, args.batch_size, True, args.num_workers, args.pin_memory)
    val_loader = build_loader(val_ds, args.batch_size, False, args.num_workers, args.pin_memory)

    x, y = next(iter(train_loader))
    print("=== Batch Check ===")
    print("train batch x:", tuple(x.shape), "y:", tuple(y.shape), "y min/max:", int(y.min()), int(y.max()))
    x2, y2 = next(iter(val_loader))
    print("val   batch x:", tuple(x2.shape), "y:", tuple(y2.shape), "y min/max:", int(y2.min()), int(y2.max()))
    print()

    # forward 체크
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyDummy(num_classes=len(train_ds.classes)).to(device)
    with torch.no_grad():
        logits = model(x.to(device))
    print("=== Forward Check ===")
    print("device:", device)
    print("input shape :", tuple(x.shape))
    print("output shape:", tuple(logits.shape))
    print()

    # (옵션) val을 45k/5k로 추가 분할해서 확인
    if args.subset_val:
        print("=== Subset Split Check (from split/val) ===")
        n = len(val_ds)

        if args.train_size + args.val_size > n:
            raise RuntimeError(
                f"subset sizes too large: train_size+val_size={args.train_size + args.val_size} > n={n}"
            )

        g = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(n, generator=g).tolist()

        tr_idx = perm[:args.train_size]
        va_idx = perm[args.train_size:args.train_size + args.val_size]

        sub_train = Subset(val_ds, tr_idx)
        sub_val = Subset(val_ds, va_idx)

        print("subset train:", len(sub_train))
        print("subset val  :", len(sub_val))

        sub_train_loader = build_loader(sub_train, args.batch_size, True, args.num_workers, args.pin_memory)
        sub_val_loader = build_loader(sub_val, args.batch_size, False, args.num_workers, args.pin_memory)

        xx, yy = next(iter(sub_train_loader))
        print("subset train batch:", tuple(xx.shape), tuple(yy.shape), int(yy.min()), int(yy.max()))
        xx2, yy2 = next(iter(sub_val_loader))
        print("subset val   batch:", tuple(xx2.shape), tuple(yy2.shape), int(yy2.min()), int(yy2.max()))
        print()

    print("Loader check finished successfully.")


if __name__ == "__main__":
    main()
