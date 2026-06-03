# REASONIX.md

## Stack

- **Python 3** + **PyTorch** (>=2.0) + **torchvision** — core ML framework
- **transformers** (>=4.35) + **datasets** (>=2.14) — HuggingFace model distillation
- **accelerate** + **bitsandbytes** — optional, for Qwen quantized teacher loading
- No web framework, no CLI framework — each script is a standalone `python <file>.py`

## Layout

| Path | What's in it |
|---|---|
| `models.py` | TeacherCNN (large) and StudentCNN (small) — both CNNs for CIFAR-10 |
| `distill.py` | `distillation_loss()`, `train_one_epoch()`, `evaluate()` — shared training utilities |
| `train_teacher.py` | Train from-scratch CNN teacher on CIFAR-10 → `teacher.pth` |
| `train_teacher_pretrained.py` | Fine-tune ImageNet-pretrained ResNet-18 on CIFAR-10 → `teacher_pretrained.pth` |
| `teacher_pretrained.py` | `build_pretrained_teacher()` — ResNet-18 adapted for 32×32 inputs |
| `train_student.py` | Distill StudentCNN from either teacher (`--teacher manual|pretrained`), plus baseline comparison |
| `distill_transformers.py` | Distill BERT-base → BERT-tiny on SST-2 sentiment classification |
| `distill_qwen.py` | Distill Qwen2.5-1.5B → Qwen2.5-0.5B on WikiText-2 (causal LM) |
| `requirements.txt` | Pip dependencies, split into core / optional Transformer / optional Qwen groups |

## Commands

```bash
# Manual CNN teacher + student (CIFAR-10)
python train_teacher.py                           # train from-scratch CNN teacher
python train_student.py --teacher manual          # distill student from it

# Pretrained ResNet-18 teacher + student (CIFAR-10)
python train_teacher_pretrained.py                # fine-tune ResNet-18
python train_student.py --teacher pretrained      # distill student from it

# HuggingFace transformer distillation (SST-2)
python distill_transformers.py

# Qwen2.5 causal LM distillation (WikiText-2)
python distill_qwen.py
```

No build step, no test runner, no lint/format commands are defined.

## Conventions

- **Config as top-level constants** — `EPOCHS`, `BATCH_SIZE`, `LR`, `TEMPERATURE`, `ALPHA`, `DEVICE`, `SAVE_PATH` in SCREAMING_SNAKE_CASE at the top of each training script.
- **Section separators** — `# ── Config ──`, `# ── Data ──`, `# ── Model ──`, etc. with Unicode box-drawing characters.
- **Device-agnostic** — every script uses `torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
- **Shared core** — `distill.py` is imported by all training scripts; it's the only shared module. Everything else is self-contained.
- **Type hints on API functions** — `distill.py` annotates parameters/returns; training scripts do not.

## Watch out for

- **No .gitignore** — generated artifacts (`.pth` checkpoints, `data/`, `__pycache__/`, `student_transformer/`, `student_qwen_distilled/`) will appear in `git status`.
- **HF mirror hardcoded** — `distill_transformers.py` sets `HF_ENDPOINT=https://hf-mirror.com` for mainland China access. Remove or change if you're outside China.
- **Teacher quantization in `distill_qwen.py`** — defaults to int4 (requires `bitsandbytes` + CUDA GPU). Set `TEACHER_QUANT = "none"` if you have enough VRAM or are on CPU.
- **CIFAR-10 mean/std duplicated** — each CIFAR-10 script redefines `CIFAR_MEAN` / `CIFAR_STD` locally. Changing normalization means touching multiple files.
