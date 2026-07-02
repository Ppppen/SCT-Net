import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from typing import List, Dict, Tuple
import glob
from pathlib import Path
import random


class RoadGeometryDataset(Dataset):

    def __init__(self,
                 image_dir: str,
                 gt_dir: str,
                 image_size: Tuple[int, int] = (512, 512),
                 augmentation: bool = True,
                 dataset_type: str = 'deepglobe'):
        super(RoadGeometryDataset, self).__init__()

        self.image_dir = image_dir
        self.gt_dir = gt_dir
        self.image_size = image_size
        self.augmentation = augmentation
        self.dataset_type = dataset_type

        self._setup_dataset_config()
        self.file_list = self._load_file_list()

        print(f"加载了 {len(self.file_list)} 个样本")
        print(f"数据集类型: {dataset_type}")
        print(f"数据增强: {augmentation}")
        print(f"图像尺寸: {image_size}")

    def _setup_dataset_config(self):
        if self.dataset_type == 'deepglobe':
            self.image_suffix = '_sat.jpg'
            self.gt_suffix = '_mask.png'
        elif self.dataset_type == 'globe_scale':
            self.image_suffix = '_sat.png'
            self.gt_suffix = '_gt.png'
        elif self.dataset_type == 'cityscale':
            self.image_suffix = '_sat.png'
            self.gt_suffix = '_mask.png'
        elif self.dataset_type == 'spacenet':
            self.image_suffix = '__rgb.png'
            self.gt_suffix = '__mask.png'
        elif self.dataset_type == 'CHIN6_CUG':
            self.image_suffix = '_sat.jpg'
            self.gt_suffix = '_mask.png'
            self.npy_dir_suffix = '_mask'
        else:
            raise ValueError(f"不支持的数据集类型: {self.dataset_type}")

    def _load_file_list(self) -> List[str]:
        image_pattern = os.path.join(self.image_dir, f"*{self.image_suffix}")
        image_files = glob.glob(image_pattern)

        file_bases = []
        for img_path in image_files:
            base_name = Path(img_path).stem
            if self.dataset_type == 'deepglobe':
                base_name = base_name.replace('_sat', '')
            elif self.dataset_type == 'spacenet':
                base_name = base_name.replace('__rgb', '')
            elif self.dataset_type == 'CHIN6_CUG':
                base_name = base_name.replace('_sat', '')

            # 仅检查 GT Mask 是否存在
            gt_path = os.path.join(self.gt_dir, f"{base_name}{self.gt_suffix}")
            if not os.path.exists(gt_path):
                continue


            file_bases.append(base_name)

        return sorted(file_bases)

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        base_name = self.file_list[idx]

        # 1. 加载图像和 Mask
        image, gt_mask = self._load_image_and_gt(base_name)


        # 2. 数据增强
        if self.augmentation:
            image, gt_mask = self._apply_augmentation(image, gt_mask)

        self.idx = idx
        # 3. 转 Tensor
        sample = self._to_tensor(image, gt_mask)

        return sample

    def _load_image_and_gt(self, base_name: str) -> Tuple[np.ndarray, np.ndarray]:
        image_path = os.path.join(self.image_dir, f"{base_name}{self.image_suffix}")
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        gt_path = os.path.join(self.gt_dir, f"{base_name}{self.gt_suffix}")
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        # Resize
        if image.shape[:2] != self.image_size:
            image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
            gt_mask = cv2.resize(gt_mask, (self.image_size[1], self.image_size[0]))

        # Normalize
        image = image.astype(np.float32) / 255.0
        gt_mask = (gt_mask > 127).astype(np.float32)

        return image, gt_mask


    def _apply_augmentation(self, image, gt_mask):
        rotation = random.choice([0, 1, 2, 3])
        if rotation == 0:
            return image, gt_mask

        image = np.rot90(image, rotation).copy()
        gt_mask = np.rot90(gt_mask, rotation).copy()


        return image, gt_mask


    def _to_tensor(self, image, gt_mask):
        """转换为PyTorch Tensor格式"""

        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)
        if not gt_mask.flags['C_CONTIGUOUS']:
            gt_mask = np.ascontiguousarray(gt_mask)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()

        gt_tensor = torch.from_numpy(gt_mask).unsqueeze(0).float()


        return {
            'image': image_tensor,
            'gt_mask': gt_tensor,
            'file_name': self.file_list[self.idx] if hasattr(self, 'idx') else ''
        }


class RoadGeometryDataLoader:
    def __init__(self,
                 image_dir: str,
                 gt_dir: str,
                 batch_size: int = 4,
                 image_size: Tuple[int, int] = (512, 512),
                 augmentation: bool = True,
                 shuffle: bool = True,
                 num_workers: int = 4,
                 dataset_type: str = 'deepglobe'):
        self.dataset = RoadGeometryDataset(
            image_dir=image_dir,
            gt_dir=gt_dir,
            image_size=image_size,
            augmentation=augmentation,
            dataset_type=dataset_type
        )

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)


if __name__ == "__main__":
    pass