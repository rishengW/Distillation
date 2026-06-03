"""
Knowledge distillation using HuggingFace transformer models.

Teacher: bert-base-uncased  (110M params)
Student: prajjwal1/bert-tiny (4.4M params)

Task: SST-2 sentiment classification (positive / negative)

Install deps:
    pip install torch transformers datasets
"""

import os
# Use HuggingFace mirror for stable access from mainland China
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from datasets import load_dataset
from distill import distillation_loss

# ── Config ────────────────────────────────────────────────────────────────────
TEACHER_NAME = "textattack/bert-base-uncased-SST-2"   # already fine-tuned on SST-2
STUDENT_NAME = "prajjwal1/bert-tiny"                  # tiny BERT, 2 layers
NUM_LABELS = 2
EPOCHS = 3
BATCH_SIZE = 32
LR = 5e-5
TEMPERATURE = 4.0
ALPHA = 0.7
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Data ──────────────────────────────────────────────────────────────────────
print("Loading SST-2 dataset...")
raw = load_dataset("glue", "sst2")
tokenizer = AutoTokenizer.from_pretrained(TEACHER_NAME)


def tokenize(batch):
    return tokenizer(batch["sentence"], truncation=True, padding="max_length", max_length=MAX_LEN)


encoded = raw.map(tokenize, batched=True)
encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

train_loader = DataLoader(encoded["train"], batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(encoded["validation"], batch_size=64, shuffle=False)

# ── Models ────────────────────────────────────────────────────────────────────
print(f"Loading teacher: {TEACHER_NAME}")
teacher = AutoModelForSequenceClassification.from_pretrained(TEACHER_NAME, num_labels=NUM_LABELS).to(DEVICE)
teacher.eval()

print(f"Loading student: {STUDENT_NAME}")
student = AutoModelForSequenceClassification.from_pretrained(
    STUDENT_NAME, num_labels=NUM_LABELS, ignore_mismatched_sizes=True
).to(DEVICE)

teacher_params = sum(p.numel() for p in teacher.parameters())
student_params = sum(p.numel() for p in student.parameters())
print(f"Teacher params: {teacher_params:,}")
print(f"Student params: {student_params:,}  ({student_params/teacher_params*100:.1f}% of teacher)\n")

# ── Optimizer & scheduler ─────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)


# ── Eval helper ───────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


# ── Teacher baseline ──────────────────────────────────────────────────────────
teacher_acc = evaluate(teacher, val_loader)
print(f"Teacher validation accuracy: {teacher_acc:.4f}\n")

# ── Distillation training loop ────────────────────────────────────────────────
print("=== Distilling student from teacher ===")
for epoch in range(1, EPOCHS + 1):
    student.train()
    total_loss = 0.0

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        student_logits = student(input_ids=input_ids, attention_mask=attention_mask).logits

        with torch.no_grad():
            teacher_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits

        loss = distillation_loss(student_logits, teacher_logits, labels, TEMPERATURE, ALPHA)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        if (step + 1) % 100 == 0:
            print(f"  step {step+1}/{len(train_loader)}  loss={loss.item():.4f}")

    val_acc = evaluate(student, val_loader)
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}/{EPOCHS}  avg_loss={avg_loss:.4f}  val_acc={val_acc:.4f}\n")

# ── Save & final report ───────────────────────────────────────────────────────
student.save_pretrained("student_transformer")
tokenizer.save_pretrained("student_transformer")

final_student_acc = evaluate(student, val_loader)
print("=== Final Results ===")
print(f"Teacher  acc: {teacher_acc:.4f}  params: {teacher_params:,}")
print(f"Student  acc: {final_student_acc:.4f}  params: {student_params:,}")
print(f"Accuracy retention: {final_student_acc/teacher_acc*100:.1f}%")
print(f"Parameter reduction: {(1 - student_params/teacher_params)*100:.1f}%")
