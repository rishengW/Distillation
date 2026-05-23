"""
Knowledge distillation between two Qwen2.5 causal language models.

Teacher : Qwen/Qwen2.5-1.5B-Instruct  (~1.5 B params)
Student : Qwen/Qwen2.5-0.5B-Instruct  (~0.5 B params)

Task    : Causal language-model distillation on the WikiText-2 dataset.
          The student learns to mimic the teacher's next-token distribution
          (soft targets via KL divergence) while also minimising standard
          cross-entropy on the ground-truth tokens (hard targets).

Loss    : L = α · T² · KL(student_soft || teacher_soft)
            + (1 - α) · CrossEntropy(student_logits, labels)

Install deps:
    pip install torch transformers datasets accelerate

Notes:
  - For GPU-poor setups the teacher is loaded in 8-bit (bitsandbytes) when
    a CUDA device is available; remove `load_in_8bit` if you have enough VRAM.
  - Gradient checkpointing is enabled on the student to save memory.
  - Mixed-precision (bf16/fp16) is used automatically when supported.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
TEACHER_NAME  = "Qwen/Qwen2.5-1.5B-Instruct"
STUDENT_NAME  = "Qwen/Qwen2.5-0.5B-Instruct"

EPOCHS        = 3
BATCH_SIZE    = 4          # reduce if OOM; increase if you have VRAM to spare
GRAD_ACCUM    = 8          # effective batch = BATCH_SIZE * GRAD_ACCUM = 32
LR            = 2e-5
MAX_LEN       = 512        # token sequence length per sample
TEMPERATURE   = 4.0        # higher → softer teacher distribution
ALPHA         = 0.7        # weight for soft (KL) loss; (1-α) for hard (CE) loss
SAVE_PATH     = "student_qwen_distilled"

DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_BF16      = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
DTYPE         = torch.bfloat16 if USE_BF16 else (torch.float16 if torch.cuda.is_available() else torch.float32)

print(f"Device : {DEVICE}  |  dtype : {DTYPE}")

# ── Tokenizer ─────────────────────────────────────────────────────────────────
# Both Qwen2.5 models share the same tokenizer vocabulary, so we load once.
print(f"\nLoading tokenizer from {TEACHER_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(TEACHER_NAME, trust_remote_code=True)

# Qwen2.5 uses <|endoftext|> as pad; set it explicitly so the collator works.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── Dataset ───────────────────────────────────────────────────────────────────
print("Loading WikiText-2 dataset ...")
raw = load_dataset("wikitext", "wikitext-2-raw-v1")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LEN,
        padding=False,          # collator handles padding
        return_special_tokens_mask=False,
    )

tokenized = raw.map(
    tokenize,
    batched=True,
    remove_columns=raw["train"].column_names,
)
tokenized.set_format("torch")

# DataCollatorForLanguageModeling pads sequences and creates -100 labels for
# padding positions so they are ignored in the cross-entropy loss.
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

train_loader = DataLoader(
    tokenized["train"],
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collator,
)
val_loader = DataLoader(
    tokenized["validation"],
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    collate_fn=collator,
)

# ── Load teacher ──────────────────────────────────────────────────────────────
print(f"\nLoading teacher : {TEACHER_NAME}")
teacher_kwargs = dict(
    torch_dtype=DTYPE,
    trust_remote_code=True,
    device_map="auto" if torch.cuda.is_available() else None,
)
teacher = AutoModelForCausalLM.from_pretrained(TEACHER_NAME, **teacher_kwargs)
teacher.eval()
# Freeze teacher — we never update its weights.
for p in teacher.parameters():
    p.requires_grad_(False)

teacher_params = sum(p.numel() for p in teacher.parameters())
print(f"  Teacher params : {teacher_params:,}")

# ── Load student ──────────────────────────────────────────────────────────────
print(f"\nLoading student : {STUDENT_NAME}")
student_kwargs = dict(
    torch_dtype=DTYPE,
    trust_remote_code=True,
    device_map="auto" if torch.cuda.is_available() else None,
)
student = AutoModelForCausalLM.from_pretrained(STUDENT_NAME, **student_kwargs)
student.train()
student.gradient_checkpointing_enable()   # trades compute for memory

student_params = sum(p.numel() for p in student.parameters())
print(f"  Student params : {student_params:,}  ({student_params / teacher_params * 100:.1f}% of teacher)\n")

# ── Distillation loss ─────────────────────────────────────────────────────────
def distillation_loss_lm(
    student_logits: torch.Tensor,   # [B, T, V]
    teacher_logits: torch.Tensor,   # [B, T, V]
    labels: torch.Tensor,           # [B, T]  (-100 at padding positions)
    temperature: float,
    alpha: float,
) -> torch.Tensor:
    """
    Combined soft + hard loss for causal LM distillation.

    Soft loss  : KL divergence between student and teacher next-token
                 distributions, computed only on non-padding positions.
    Hard loss  : Standard cross-entropy against ground-truth tokens
                 (padding positions already masked via -100 labels).
    """
    B, T, V = student_logits.shape

    # ── Soft loss (KL divergence) ─────────────────────────────────────────────
    # Work in float32 for numerical stability regardless of model dtype.
    s_logits = student_logits.float()
    t_logits = teacher_logits.float()

    soft_student = F.log_softmax(s_logits / temperature, dim=-1)   # [B, T, V]
    soft_teacher = F.softmax(t_logits / temperature, dim=-1)        # [B, T, V]

    # Mask out padding positions (label == -100).
    mask = (labels != -100).float()                                  # [B, T]

    # KL per token, then average over non-padding tokens.
    kl_per_token = F.kl_div(
        soft_student.view(-1, V),
        soft_teacher.view(-1, V),
        reduction="none",
    ).sum(dim=-1)                                                    # [B*T]

    kl_per_token = kl_per_token.view(B, T)
    soft_loss = (kl_per_token * mask).sum() / mask.sum().clamp(min=1)
    soft_loss = soft_loss * (temperature ** 2)

    # ── Hard loss (cross-entropy) ─────────────────────────────────────────────
    # Shift so that token i predicts token i+1 (standard causal LM setup).
    shift_logits = student_logits[..., :-1, :].contiguous().float()  # [B, T-1, V]
    shift_labels = labels[..., 1:].contiguous()                       # [B, T-1]
    hard_loss = F.cross_entropy(
        shift_logits.view(-1, V),
        shift_labels.view(-1),
        ignore_index=-100,
    )

    return alpha * soft_loss + (1.0 - alpha) * hard_loss


# ── Perplexity evaluation ─────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_perplexity(model, loader, max_batches: int = 50) -> float:
    """
    Compute perplexity on up to `max_batches` batches.
    Lower is better; a random model over 50k vocab ≈ 50,000.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(DEVICE)
        labels    = batch["labels"].to(DEVICE)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(DEVICE)

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # out.loss is mean CE over non-padding tokens
        n_tokens = (labels != -100).sum().item()
        total_loss   += out.loss.item() * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    return float(torch.exp(torch.tensor(avg_loss)))


# ── Optimizer & scheduler ─────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(
    student.parameters(),
    lr=LR,
    weight_decay=0.01,
    betas=(0.9, 0.95),
)
total_steps    = (len(train_loader) // GRAD_ACCUM) * EPOCHS
warmup_steps   = total_steps // 10
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

# ── Baseline perplexity ───────────────────────────────────────────────────────
print("=== Baseline perplexity (before distillation) ===")
teacher_ppl = evaluate_perplexity(teacher, val_loader)
student_ppl = evaluate_perplexity(student, val_loader)
print(f"  Teacher PPL : {teacher_ppl:.2f}")
print(f"  Student PPL : {student_ppl:.2f}\n")

# ── Distillation training loop ────────────────────────────────────────────────
print("=== Starting distillation ===")
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available() and not USE_BF16)

for epoch in range(1, EPOCHS + 1):
    student.train()
    total_loss   = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        input_ids      = batch["input_ids"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(DEVICE)

        # ── Forward passes ────────────────────────────────────────────────────
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=DTYPE):
            student_out = student(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            with torch.no_grad():
                teacher_out = teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

            loss = distillation_loss_lm(
                student_out.logits,
                teacher_out.logits,
                labels,
                TEMPERATURE,
                ALPHA,
            )
            loss = loss / GRAD_ACCUM   # scale for gradient accumulation

        scaler.scale(loss).backward()
        total_loss += loss.item() * GRAD_ACCUM   # un-scale for logging

        # ── Gradient accumulation step ────────────────────────────────────────
        if (step + 1) % GRAD_ACCUM == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        if (step + 1) % 100 == 0:
            avg = total_loss / (step + 1)
            print(f"  Epoch {epoch} | step {step+1}/{len(train_loader)} | loss={avg:.4f}")

    # ── End-of-epoch eval ─────────────────────────────────────────────────────
    val_ppl  = evaluate_perplexity(student, val_loader)
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}/{EPOCHS}  avg_loss={avg_loss:.4f}  val_ppl={val_ppl:.2f}\n")

# ── Save distilled student ────────────────────────────────────────────────────
student.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print(f"Distilled student saved to '{SAVE_PATH}/'")

# ── Final report ──────────────────────────────────────────────────────────────
final_ppl = evaluate_perplexity(student, val_loader)
print("\n=== Final Results ===")
print(f"Teacher  PPL : {teacher_ppl:.2f}   params : {teacher_params:,}")
print(f"Student  PPL : {final_ppl:.2f}   params : {student_params:,}")
print(f"PPL gap      : {final_ppl - teacher_ppl:+.2f}")
print(f"Param reduction : {(1 - student_params / teacher_params) * 100:.1f}%")
