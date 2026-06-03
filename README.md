# Knowledge Distillation

Train a small **student** model to mimic a large **teacher** model using soft targets (logits), not just hard labels.

## Two Modes

### 1. Manual Models (CIFAR-10, no pretrained weights needed)
A large CNN teacher and a small CNN student trained on CIFAR-10.

You can choose between two teachers:

**Option A — From-scratch CNN teacher**
```bash
pip install torch torchvision
python train_teacher.py                          # train & save TeacherCNN
python train_student.py --teacher manual         # distill student
```

**Option B — Pretrained ResNet-18 teacher (torchvision)**
```bash
pip install torch torchvision
python train_teacher_pretrained.py               # fine-tune ResNet-18
python train_student.py --teacher pretrained     # distill student
```

### 2. HuggingFace Transformer Models
Distill a large transformer (e.g. BERT-base) into a smaller one (e.g. DistilBERT or a tiny BERT).

```bash
pip install torch transformers datasets
export HF_ENDPOINT=https://hf-mirror.com # set the mirror
python distill_transformers.py
```

## How It Works

The distillation loss combines:
- **Soft loss**: KL divergence between teacher and student output distributions (scaled by temperature T)
- **Hard loss**: standard cross-entropy against ground truth labels
- **Final loss**: `alpha * soft_loss + (1 - alpha) * hard_loss`

Higher temperature T produces softer probability distributions, revealing more of the teacher's "dark knowledge".
