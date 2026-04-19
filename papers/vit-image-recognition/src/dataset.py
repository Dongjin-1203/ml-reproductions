"""
dataset.py
----------
CIFAR-10 데이터셋 로딩 및 전처리 모듈.

torchvision을 사용하여 CIFAR-10을 다운로드하고,
train/val/test로 분할한 뒤 augmentation이 적용된 DataLoader를 반환한다.
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_transforms(image_size: int, train: bool) -> transforms.Compose:
    """train/eval 용 transform 반환.

    Args:
        image_size: 입력 이미지 크기 (CIFAR-10은 32).
        train: True면 augmentation 포함, False면 normalize만 적용.

    Returns:
        torchvision transforms.Compose 객체.
    """
    # CIFAR-10 채널별 mean/std (ImageNet 값 아님)
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    if train:
        return transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def get_dataloaders(
    config_path: str | Path,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """config.yaml을 읽어 train/val/test DataLoader를 반환한다.

    Args:
        config_path: config.yaml 파일 경로.

    Returns:
        (train_loader, val_loader, test_loader) 튜플.
    """
    cfg = load_config(config_path)
    ds_cfg = cfg["dataset"]
    tr_cfg = cfg["training"]

    data_dir   = ds_cfg["data_dir"]
    image_size = ds_cfg["image_size"]
    val_split  = ds_cfg["val_split"]
    batch_size = tr_cfg["batch_size"]
    num_workers = ds_cfg["num_workers"]

    # Train dataset (augmentation 포함)
    train_full = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=get_transforms(image_size, train=True),
    )

    # Val split: train_full에서 val_split 비율만큼 분리
    n_total = len(train_full)
    n_val   = int(n_total * val_split)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(tr_cfg["seed"])
    train_dataset, val_dataset = random_split(
        train_full, [n_train, n_val], generator=generator
    )

    # Val dataset은 augmentation 없이 별도 transform 적용
    # random_split은 원본 dataset의 transform을 공유하므로
    # Subset을 래핑하여 eval transform을 덮어씀
    val_dataset = _SubsetWithTransform(
        train_full,
        val_dataset.indices,
        transform=get_transforms(image_size, train=False),
    )

    # Test dataset (augmentation 없음)
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=get_transforms(image_size, train=False),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


class _SubsetWithTransform(torch.utils.data.Dataset):
    """Subset에 별도 transform을 적용하기 위한 래퍼 클래스."""

    def __init__(
        self,
        dataset: datasets.CIFAR10,
        indices: list[int],
        transform: transforms.Compose,
    ) -> None:
        self.dataset   = dataset
        self.indices   = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        img, label = self.dataset.data[self.indices[idx]], int(self.dataset.targets[self.indices[idx]])
        # dataset.data는 numpy HWC uint8 → PIL로 변환 후 transform 적용
        from PIL import Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label
