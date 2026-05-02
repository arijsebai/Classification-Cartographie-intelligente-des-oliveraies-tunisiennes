from __future__ import annotations

from typing import Iterable, Sequence, Tuple


def build_model(encoder_name: str = "resnet34", in_channels: int = 5, classes: int = 1, encoder_weights: str = "imagenet"):
    try:
        import segmentation_models_pytorch as smp
    except Exception as exc:  # pragma: no cover - runtime dependency guard
        # Provide a lightweight UNet fallback implemented with PyTorch to avoid
        # requiring the heavy `segmentation-models-pytorch` package for quick
        # smoke tests on machines without the full toolchain.
        try:
            import torch
            import torch.nn as nn

            class _ConvBlock(nn.Module):
                def __init__(self, in_ch, out_ch):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                    )

                def forward(self, x):
                    return self.net(x)

            class _UNetSmall(nn.Module):
                def __init__(self, in_ch, out_ch):
                    super().__init__()
                    self.enc1 = _ConvBlock(in_ch, 32)
                    self.enc2 = _ConvBlock(32, 64)
                    self.enc3 = _ConvBlock(64, 128)
                    self.pool = nn.MaxPool2d(2)
                    self.up = nn.Upsample(scale_factor=2, mode="nearest")
                    self.dec2 = _ConvBlock(128 + 64, 64)
                    self.dec1 = _ConvBlock(64 + 32, 32)
                    self.head = nn.Conv2d(32, out_ch, kernel_size=1)

                def forward(self, x):
                    e1 = self.enc1(x)
                    e2 = self.enc2(self.pool(e1))
                    e3 = self.enc3(self.pool(e2))
                    d2 = self.up(e3)
                    d2 = torch.cat([d2, e2], dim=1)
                    d2 = self.dec2(d2)
                    d1 = self.up(d2)
                    d1 = torch.cat([d1, e1], dim=1)
                    d1 = self.dec1(d1)
                    return self.head(d1)

            return _UNetSmall(in_ch=in_channels, out_ch=classes)
        except Exception:
            raise RuntimeError("segmentation_models_pytorch is required and fallback failed") from exc

    # Preferred path: use smp when available
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )


def to_tensor_batch(images, masks):
    try:
        import torch
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torch and numpy are required") from exc

    images = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous().float()
    masks = torch.from_numpy(np.stack(masks))[:, None, :, :].contiguous().float()
    return images, masks


def binary_dice_loss(logits, targets, eps: float = 1e-6):
    import torch

    probs = torch.sigmoid(logits)
    intersection = torch.sum(probs * targets)
    denom = torch.sum(probs) + torch.sum(targets)
    return 1.0 - ((2.0 * intersection + eps) / (denom + eps))


def binary_iou_from_logits(logits, targets, threshold: float = 0.5, eps: float = 1e-6):
    import torch

    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = (targets > 0.5).float()
    intersection = torch.sum(preds * targets)
    union = torch.sum(preds) + torch.sum(targets) - intersection
    return (intersection + eps) / (union + eps)
