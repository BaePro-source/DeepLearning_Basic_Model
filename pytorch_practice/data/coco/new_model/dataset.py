"""
COCO Semantic Segmentation Dataset
"""
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

import torchvision.transforms as T
import torchvision.transforms.functional as TF


class COCOSemSeg(Dataset):
    """COCO instance annotations -> semantic segmentation mask"""
    
    def __init__(self, img_dir, ann_json, resize=512, crop=512, train=True, seed=42):
        self.img_dir = Path(img_dir)
        self.coco = COCO(str(ann_json))
        self.img_ids = sorted(self.coco.getImgIds())
        self.train = train
        self.resize = resize
        self.crop = crop

        # Category mapping: COCO id -> contiguous id (1..K)
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.cat_ids = sorted([c["id"] for c in cats])
        self.cat2contig = {cat_id: (i + 1) for i, cat_id in enumerate(self.cat_ids)}
        
        random.seed(seed)

    def __len__(self):
        return len(self.img_ids)

    def _anns_to_mask(self, img_info):
        """Convert COCO annotations to semantic mask"""
        h, w = img_info["height"], img_info["width"]
        mask = np.zeros((h, w), dtype=np.uint8)

        ann_ids = self.coco.getAnnIds(imgIds=img_info["id"], iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)

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

            m = maskUtils.decode(rle)
            mask[m == 1] = contig

        return mask

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        img_path = self.img_dir / img_info["file_name"]

        image = Image.open(img_path).convert("RGB")
        mask_np = self._anns_to_mask(img_info)
        mask = Image.fromarray(mask_np, mode="L")

        # Resize
        image = TF.resize(image, [self.resize, self.resize], interpolation=Image.BILINEAR)
        mask = TF.resize(mask, [self.resize, self.resize], interpolation=Image.NEAREST)

        if self.train:
            # Random crop
            i, j, h, w = T.RandomCrop.get_params(image, output_size=(self.crop, self.crop))
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

            # Random horizontal flip
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
        else:
            # Center crop for validation
            image = TF.center_crop(image, [self.crop, self.crop])
            mask = TF.center_crop(mask, [self.crop, self.crop])

        # To tensor
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mask = torch.from_numpy(np.array(mask, dtype=np.uint8)).long()

        return image, mask