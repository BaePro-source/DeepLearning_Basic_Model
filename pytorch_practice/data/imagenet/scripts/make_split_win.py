from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List


IMG_EXTS = {".jpeg", ".jpg", ".png"}  # .JPEG 포함됨(소문자 비교)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_place(src: Path, dst: Path, mode: str) -> str:
    """
    mode == "hardlink": os.link 시도, 실패하면 copy2로 fallback
    mode == "copy": copy2
    return: "hardlink" or "copy"
    """
    if dst.exists():
        return "skip"

    ensure_dir(dst.parent)

    if mode == "copy":
        shutil.copy2(src, dst)
        return "copy"

    # hardlink 시도
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        # 다른 파일시스템(cross-device) 등으로 hardlink 실패 시 copy로 fallback
        shutil.copy2(src, dst)
        return "copy"


def resolve_default_paths() -> Dict[str, Path]:
    """
    스크립트를 어디서 실행하든, 현재 파일 위치 기준으로 경로를 잡습니다.
    (프로젝트 구조 예시)
    pytorch_practice/
      data/imagenet/
        data/
          raw/ILSVRC2012_img_val
          labels/ILSVRC2012_validation_ground_truth.txt
          meta/imagenet_class_index.json
          split/
        scripts/
          (이 스크립트 위치 가능)
    """
    script_dir = Path(__file__).resolve().parent
    imagenet_root = script_dir.parent  # .../data/imagenet (scripts의 부모)
    data_dir = imagenet_root / "data"
    return {
        "VAL_DIR": data_dir / "raw" / "ILSVRC2012_img_val",
        "GT_FILE": data_dir / "labels" / "ILSVRC2012_validation_ground_truth.txt",
        "CLASS_INDEX_JSON": data_dir / "meta" / "imagenet_class_index.json",
        "OUT_DIR": data_dir / "split",
    }


def load_class_index(class_index_json: Path) -> Dict[int, str]:
    # imagenet_class_index.json 형식: {"0": ["n01440764", "tench"], ...}
    obj = json.loads(class_index_json.read_text(encoding="utf-8"))
    return {int(k): v[0] for k, v in obj.items()}


def load_gt_labels(gt_file: Path) -> List[int]:
    # GT는 1-based class id (1~1000)인 경우가 일반적
    lines = [ln.strip() for ln in gt_file.read_text().splitlines() if ln.strip()]
    return [int(x) for x in lines]


def list_val_images(val_dir: Path) -> List[Path]:
    imgs = []
    for p in val_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            imgs.append(p)
    # ImageNet val은 파일명이 ILSVRC2012_val_00000001.JPEG 형태라 정렬이 중요
    return sorted(imgs, key=lambda x: x.name)


def main() -> None:
    paths = resolve_default_paths()

    parser = argparse.ArgumentParser()
    parser.add_argument("--val_dir", type=str, default=str(paths["VAL_DIR"]))
    parser.add_argument("--gt_file", type=str, default=str(paths["GT_FILE"]))
    parser.add_argument("--class_index_json", type=str, default=str(paths["CLASS_INDEX_JSON"]))
    parser.add_argument("--out_dir", type=str, default=str(paths["OUT_DIR"]))

    parser.add_argument("--train_per_class", type=int, default=45)
    parser.add_argument("--val_per_class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--mode", choices=["hardlink", "copy"], default="hardlink",
                        help="hardlink(권장, 용량 거의 증가 없음). 실패 시 자동 copy fallback")
    parser.add_argument("--force", action="store_true",
                        help="기존 out_dir/train, out_dir/val이 있으면 삭제 후 재생성")

    args = parser.parse_args()

    VAL_DIR = Path(args.val_dir)
    GT_FILE = Path(args.gt_file)
    CLASS_INDEX_JSON = Path(args.class_index_json)
    OUT_DIR = Path(args.out_dir)

    train_per = args.train_per_class
    val_per = args.val_per_class
    seed = args.seed
    mode = args.mode

    print("=== Config ===")
    print("VAL_DIR          :", VAL_DIR)
    print("GT_FILE          :", GT_FILE)
    print("CLASS_INDEX_JSON :", CLASS_INDEX_JSON)
    print("OUT_DIR          :", OUT_DIR)
    print("train_per_class  :", train_per)
    print("val_per_class    :", val_per)
    print("seed             :", seed)
    print("mode             :", mode)
    print("force            :", args.force)
    print()

    if not VAL_DIR.is_dir():
        raise RuntimeError(f"VAL_DIR not found or not a dir: {VAL_DIR}")
    if not GT_FILE.is_file():
        raise RuntimeError(f"GT_FILE not found: {GT_FILE}")
    if not CLASS_INDEX_JSON.is_file():
        raise RuntimeError(f"CLASS_INDEX_JSON not found: {CLASS_INDEX_JSON}")

    class_map = load_class_index(CLASS_INDEX_JSON)
    if len(class_map) != 1000:
        raise RuntimeError(f"class_index_json should map 1000 classes, got {len(class_map)}")

    imgs = list_val_images(VAL_DIR)
    if len(imgs) != 50000:
        raise RuntimeError(f"Expected 50000 val images in raw, got {len(imgs)} (dir={VAL_DIR})")

    gt = load_gt_labels(GT_FILE)
    if len(gt) != 50000:
        raise RuntimeError(f"Expected 50000 labels in GT, got {len(gt)} (file={GT_FILE})")

    # 출력 폴더 준비
    train_root = OUT_DIR / "train"
    val_root = OUT_DIR / "val"

    if args.force:
        if train_root.exists():
            shutil.rmtree(train_root)
        if val_root.exists():
            shutil.rmtree(val_root)

    ensure_dir(train_root)
    ensure_dir(val_root)

    # 클래스별 파일 모으기
    per_class: Dict[int, List[Path]] = {c: [] for c in range(1000)}
    for img_path, label_1based in zip(imgs, gt):
        c = int(label_1based) - 1
        if not (0 <= c < 1000):
            raise RuntimeError(f"GT label out of range: {label_1based}")
        per_class[c].append(img_path)

    # 간단 검증: ImageNet val은 클래스당 50장
    for c in range(1000):
        if len(per_class[c]) == 0:
            raise RuntimeError(f"class {c} has 0 images (GT mismatch?)")
        need = train_per + val_per
        if len(per_class[c]) < need:
            raise RuntimeError(f"class {c} needs {need} but has {len(per_class[c])}")

    random.seed(seed)

    hardlink_cnt = 0
    copy_cnt = 0
    skip_cnt = 0

    print("=== Splitting & Placing Files ===")
    for c in range(1000):
        wnid = class_map[c]
        files = per_class[c]
        random.shuffle(files)

        train_files = files[:train_per]
        val_files = files[train_per:train_per + val_per]

        # 진행 표시(너무 시끄럽지 않게 50클래스마다)
        if c % 50 == 0:
            print(f"  class {c:04d}/0999 ({wnid}) ...")

        for src in train_files:
            res = safe_place(src, train_root / wnid / src.name, mode)
            if res == "hardlink":
                hardlink_cnt += 1
            elif res == "copy":
                copy_cnt += 1
            else:
                skip_cnt += 1

        for src in val_files:
            res = safe_place(src, val_root / wnid / src.name, mode)
            if res == "hardlink":
                hardlink_cnt += 1
            elif res == "copy":
                copy_cnt += 1
            else:
                skip_cnt += 1

    print()
    print("=== Done ===")
    print("Placed (hardlink):", hardlink_cnt)
    print("Placed (copy)    :", copy_cnt)
    print("Skipped(existing):", skip_cnt)
    print("OUT_DIR          :", OUT_DIR)
    print("train_root       :", train_root)
    print("val_root         :", val_root)
    print()
    print("다음 체크:")
    print(f"  python {OUT_DIR.parent / 'scripts' / 'check_imagenet_loader.py'} --batch_size 8 --num_workers 0")
    print("완료되었습니다.")


if __name__ == "__main__":
    main()
