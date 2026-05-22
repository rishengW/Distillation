"""
Train the large teacher CNN on CIFAR-10 and save its weights.
Run this first before train_student.py.
"""

import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models import TeacherCNN
from distill import train_one_epoch, evaluate

# ── Config ────────────────────────────────────────────────────────────────────
EPOCHS = 20
BATCH_SIZE = 128
LR = 1e-3
SAVE_PATH = "teacher.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR-10 channel-wise mean/std
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

# ── Data ──────────────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

train_set = datasets.CIFAR10("data", train=True, download=True, transform=train_transform)
test_set = datasets.CIFAR10("data", train=False, download=True, transform=test_transform)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=2)

# ── Model ─────────────────────────────────────────────────────────────────────
teacher = TeacherCNN().to(DEVICE)
optimizer = optim.Adam(teacher.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

print(f"Teacher parameters: {sum(p.numel() for p in teacher.parameters()):,}")
print(f"Training on: {DEVICE}\n")

# ── Training loop ─────────────────────────────────────────────────────────────
for epoch in range(1, EPOCHS + 1):
    loss = train_one_epoch(teacher, train_loader, optimizer, DEVICE)
    acc = evaluate(teacher, test_loader, DEVICE)
    scheduler.step()
    print(f"Epoch {epoch:02d}/{EPOCHS}  loss={loss:.4f}  test_acc={acc:.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
torch.save(teacher.state_dict(), SAVE_PATH)
print(f"\nTeacher saved to {SAVE_PATH}")
