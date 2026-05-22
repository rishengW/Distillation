"""
Pretrained teacher model option using torchvision's ResNet-18.

Loads ImageNet-pretrained weights and adapts the network for CIFAR-10:
- Replaces the first conv (7x7 stride 2) with a 3x3 stride 1 conv,
  since CIFAR images (32x32) are too small for the original stem.
- Removes the initial maxpool for the same reason.
- Replaces the final FC layer with a 10-class head.

Use this as an alternative to TeacherCNN in models.py when you want
a stronger teacher without designing one from scratch.
"""

import torch.nn as nn
from torchvision import models


def build_pretrained_teacher(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """
    Return a ResNet-18 adapted for CIFAR-10.

    Args:
        num_classes: number of output classes.
        pretrained:  if True, load ImageNet-pretrained weights.
                     Set to False to train from random init.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    # Adapt stem for 32x32 inputs
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    # Replace classifier head
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


if __name__ == "__main__":
    # Quick sanity check + param count
    import torch

    m = build_pretrained_teacher()
    n_params = sum(p.numel() for p in m.parameters())
    print(f"Pretrained teacher (ResNet-18) parameters: {n_params:,}")

    x = torch.randn(2, 3, 32, 32)
    y = m(x)
    print(f"Output shape: {tuple(y.shape)}")  # expect (2, 10)
