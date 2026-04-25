"""
src/data/transforms.py — Augmentation Pipelines
================================================

Defines train and val/test transform pipelines for 224×224 face crops.

Train pipeline:
    - RandomHorizontalFlip
    - ColorJitter (brightness, contrast, saturation, hue)
    - RandomRotation ±10°
    - GaussianBlur (p=0.2)
    - ToTensor + Normalize (ImageNet stats)

Val/Test pipeline:
    - CenterCrop (guard against slight size variance)
    - ToTensor + Normalize

Usage
-----
    from src.data.transforms import get_transforms
    train_tf = get_transforms("train", image_size=224)
    val_tf   = get_transforms("val",   image_size=224)
"""

from torchvision import transforms

# ImageNet mean / std — standard for EfficientNet-B4 pretrained weights
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def get_transforms(split: str, image_size: int = 224) -> transforms.Compose:
    """Return the augmentation pipeline for a given *split*.

    Parameters
    ----------
    split : str
        One of ``"train"``, ``"val"``, or ``"test"``.
    image_size : int
        Target spatial dimension (square crop).

    Returns
    -------
    torchvision.transforms.Compose
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size + 16, image_size + 16)),   # slight oversize for crop
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.RandomRotation(degrees=10),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))],
                p=0.2,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ])

    # val / test — deterministic
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])
