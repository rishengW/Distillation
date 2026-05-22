"""
Distill the small student CNN from a saved teacher.
Also trains a baseline student (no distillation) for comparison.

Two teacher options:
  --teacher manual      → use TeacherCNN (run train_teacher.py first)
  --teacher pretrained  → use ResNet-18  (run train_teacher_pretrained.py first)
"""

import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models import TeacherCNN, StudentCNN
from teacher_pretrained import build_pretrained_teacher
from distill import train_one_epoch, evaluate

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--teacher",
    choices=["manual", "pretrained"],
    default="manual",
    help="Which teacher to distill from.",
)
args = parser.parse_args()

# ── Config ────────────────────────────────────────────────────────────────────
EPOCHS = 20
BATCH_SIZE = 128
LR = 1e-3
TEMPERATURE = 4.0   # higher = softer teacher distribution
ALPHA = 0.7         # weight for soft (distillation) loss
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.teacher == "manual":
    TEACHER_PATH = "teacher.pth"
    build_teacher = lambda: TeacherCNN()
else:
    TEACHER_PATH = "teacher_pretrained.pth"
    # pretrained=False here: weights are loaded from the fine-tuned checkpoint
    build_teacher = lambda: build_pretrained_teacher(num_classes=10, pretrained=False)

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

# ── Load teacher ──────────────────────────────────────────────────────────────
print(f"Using teacher: {args.teacher}  (checkpoint: {TEACHER_PATH})")
teacher = build_teacher().to(DEVICE)
teacher.load_state_dict(torch.load(TEACHER_PATH, map_location=DEVICE))
teacher.eval()
teacher_acc = evaluate(teacher, test_loader, DEVICE)
print(f"Teacher accuracy: {teacher_acc:.4f}")
print(f"Teacher parameters: {sum(p.numel() for p in teacher.parameters()):,}\n")

# ── Student with distillation ─────────────────────────────────────────────────
print("=== Training student WITH distillation ===")
student_distill = StudentCNN().to(DEVICE)
optimizer_d = optim.Adam(student_distill.parameters(), lr=LR)
print(f"Student parameters: {sum(p.numel() for p in student_distill.parameters()):,}")

for epoch in range(1, EPOCHS + 1):
    loss = train_one_epoch(
        student_distill, train_loader, optimizer_d, DEVICE,
        teacher=teacher, temperature=TEMPERATURE, alpha=ALPHA,
    )
    acc = evaluate(student_distill, test_loader, DEVICE)
    print(f"Epoch {epoch:02d}/{EPOCHS}  loss={loss:.4f}  test_acc={acc:.4f}")

torch.save(student_distill.state_dict(), f"student_distilled_{args.teacher}.pth")

# ── Baseline student (no distillation) ───────────────────────────────────────
print("\n=== Training student WITHOUT distillation (baseline) ===")
student_base = StudentCNN().to(DEVICE)
optimizer_b = optim.Adam(student_base.parameters(), lr=LR)

for epoch in range(1, EPOCHS + 1):
    loss = train_one_epoch(student_base, train_loader, optimizer_b, DEVICE)
    acc = evaluate(student_base, test_loader, DEVICE)
    print(f"Epoch {epoch:02d}/{EPOCHS}  loss={loss:.4f}  test_acc={acc:.4f}")

torch.save(student_base.state_dict(), "student_baseline.pth")

# ── Final comparison ──────────────────────────────────────────────────────────
print("\n=== Final Comparison ===")
print(f"Teacher          acc: {teacher_acc:.4f}  params: {sum(p.numel() for p in teacher.parameters()):,}")
print(f"Student (distill) acc: {evaluate(student_distill, test_loader, DEVICE):.4f}  params: {sum(p.numel() for p in student_distill.parameters()):,}")
print(f"Student (baseline) acc: {evaluate(student_base, test_loader, DEVICE):.4f}  params: {sum(p.numel() for p in student_base.parameters()):,}")
