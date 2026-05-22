"""
Core distillation training loop.
Works for any teacher/student pair that outputs logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.7,
) -> torch.Tensor:
    """
    Compute the combined distillation loss.

    Args:
        student_logits: raw logits from the student model  [B, C]
        teacher_logits: raw logits from the teacher model  [B, C]
        labels:         ground-truth class indices          [B]
        temperature:    T — higher = softer distributions
        alpha:          weight for soft loss (1-alpha for hard loss)

    Returns:
        Scalar loss tensor.
    """
    # Soft targets: scale logits by temperature before softmax
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)

    # KL divergence: how far student is from teacher's distribution
    # Multiply by T^2 to keep gradient magnitudes consistent across temperatures
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature ** 2)

    # Hard targets: standard cross-entropy against ground truth
    hard_loss = F.cross_entropy(student_logits, labels)

    return alpha * soft_loss + (1.0 - alpha) * hard_loss


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    teacher: nn.Module = None,
    temperature: float = 4.0,
    alpha: float = 0.7,
) -> float:
    """
    Train for one epoch.
    If teacher is provided, uses distillation loss; otherwise plain cross-entropy.
    """
    model.train()
    if teacher is not None:
        teacher.eval()

    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        student_logits = model(images)

        if teacher is not None:
            with torch.no_grad():
                teacher_logits = teacher(images)
            loss = distillation_loss(student_logits, teacher_logits, labels, temperature, alpha)
        else:
            loss = F.cross_entropy(student_logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> float:
    """Return accuracy (0–1) on the given data loader."""
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total
