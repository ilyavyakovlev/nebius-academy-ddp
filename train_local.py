import os
from itertools import chain

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # ── Model & tokenizer ──────────────────────────────────────────────
    # opt-125m is the smallest in the OPT family (~250 MB in fp16), fits easily in 4 GB VRAM.
    # Change MODEL_ID to experiment with other small models (e.g. distilgpt2).
    MODEL_ID = os.environ.get("MODEL_ID", "facebook/opt-125m")
    print(f"Loading model: {MODEL_ID}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.config.pad_token_id = tokenizer.pad_token_id

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    # Don't move model manually — Trainer/accelerate handles device placement.

    # ── Dataset ────────────────────────────────────────────────────────
    # Same dataset as train.py for easy comparison.
    train_dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")
    eval_dataset  = load_dataset("wikitext", "wikitext-2-v1", split="validation")

    # Smaller block size and batch to stay within 4 GB VRAM.
    block_size = int(os.environ.get("BLOCK_SIZE", 128))
    per_device_train_batch_size = int(os.environ.get("PER_DEVICE_TRAIN_BATCH_SIZE", 2))
    per_device_eval_batch_size  = int(os.environ.get("PER_DEVICE_EVAL_BATCH_SIZE", 2))
    gradient_accumulation_steps = int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", 4))

    def tokenize(examples):
        texts = [t for t in examples["text"] if t and not t.isspace()]
        return tokenizer(texts)

    def group_texts(examples):
        concatenated = {k: list(chain.from_iterable(examples[k])) for k in examples.keys()}
        total_length = (len(concatenated["input_ids"]) // block_size) * block_size
        if total_length == 0:
            return {k: [] for k in concatenated.keys()}
        return {
            k: [v[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, v in concatenated.items()
        }

    tokenized_train = train_dataset.map(tokenize, batched=True, remove_columns=["text"],
                                        desc="Tokenizing train")
    tokenized_train = tokenized_train.map(group_texts, batched=True,
                                          desc=f"Packing into {block_size}-token blocks")

    tokenized_eval = eval_dataset.map(tokenize, batched=True, remove_columns=["text"],
                                      desc="Tokenizing validation")
    tokenized_eval = tokenized_eval.map(group_texts, batched=True,
                                        desc=f"Packing validation into {block_size}-token blocks")

    # ── Training ───────────────────────────────────────────────────────
    args = TrainingArguments(
        output_dir="./output",
        max_steps=20,                           # smoke test — raise to 500+ for real training
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=2e-5,
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        bf16=device == "cuda",                  # RTX 3050 is Ampere — bf16 avoids GradScaler issues
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=100,
        dataloader_num_workers=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    print("Training complete.")

if __name__ == "__main__":
    main()
