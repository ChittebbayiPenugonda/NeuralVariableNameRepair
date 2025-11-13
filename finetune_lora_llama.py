#!/usr/bin/env python3
"""
Fine-tune Llama-3 Instruct with LoRA on JSONL mapping data.

Dataset JSONL rows (one per line), e.g.:
{"file":"row_000001","func_name":"id","input_text":"... <ID_1> ...","target_text":"{\"<ID_1>\": \"id\"}"}

Train target is ONLY the JSON mapping in target_text.

Usage (PACE Jupyter cell or terminal):
  python finetune_lora_llama.py \
    --data_path ./your_dataset.jsonl \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --out_dir ./lora-llama3-mapping \
    --epochs 3 --lr 2e-4 --max_len 1024

Requires: transformers, peft, datasets, accelerate, bitsandbytes, torch (CUDA)
"""

import os
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Any

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


# ---------------------------
# Helpers
# ---------------------------

SYSTEM_PROMPT = (
    "You are a coder that maps placeholder identifiers in C/C++ code to clear variable names.\n"
    "Only output a valid JSON object mapping placeholders to names, e.g.:\n"
    "{\"<ID_1>\": \"x\", \"<ID_2>\": \"count\"}\n"
    "Do not add any extra text."
)


def build_messages(user_input: str, assistant_output: str) -> List[Dict[str, str]]:
    """Build chat-style messages for llama chat template."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": assistant_output},
    ]


@dataclass
class SupervisedDataCollator:
    """
    Collator that:
    - Uses chat template to create input_ids/labels
    - Masks non-assistant tokens with -100 (so we only train on assistant JSON)
    """
    tokenizer: AutoTokenizer
    max_len: int = 1024

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []
        attention_mask_list = []

        for ex in batch:
            inp = ex["input_text"]
            tgt = ex["target_text"]

            # 1) Prompt-only ids (system + user, up to assistant start)
            prompt_msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": inp},
            ]
            prompt_text = self.tokenizer.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = self.tokenizer(
                prompt_text, add_special_tokens=False
            )["input_ids"]

            # 2) Full ids (system + user + assistant)
            full_msgs = build_messages(inp, tgt)
            full_text = self.tokenizer.apply_chat_template(
                full_msgs, tokenize=False, add_generation_prompt=False
            )
            tokenized = self.tokenizer(
                full_text,
                max_length=self.max_len,
                truncation=True,
                padding=False,
                add_special_tokens=False,
            )
            full_ids = tokenized["input_ids"]
            attn_mask = tokenized["attention_mask"]

            # If truncation removed the assistant completely, fallback to last max_len tokens
            # (still mask prompt portion correctly below).
            if len(full_ids) > self.max_len:
                full_ids = full_ids[-self.max_len:]
                attn_mask = attn_mask[-self.max_len:]

            # Determine assistant span by locating prompt_ids inside full_ids:
            # We assume full_ids begins with prompt_ids then assistant tokens.
            prompt_len = min(len(prompt_ids), len(full_ids))
            # If prompt_ids longer than full_ids due to truncation, clip:
            # Ensure matching prefix length where possible.
            # Simple safe fallback: treat the assistant start as prompt_len
            # (best-effort; in practice choose max_len large enough).
            labels = [-100] * len(full_ids)
            # Unmask only tokens after the prompt portion:
            for i in range(prompt_len, len(full_ids)):
                labels[i] = full_ids[i]

            input_ids_list.append(full_ids)
            labels_list.append(labels)
            attention_mask_list.append(attn_mask)

        # Pad to batch max
        batch_input = self.tokenizer.pad(
            {"input_ids": input_ids_list, "attention_mask": attention_mask_list},
            padding=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        # Pad labels manually with -100
        max_len = batch_input["input_ids"].size(1)
        padded_labels = []
        for lab in labels_list:
            if len(lab) < max_len:
                lab = lab + ([-100] * (max_len - len(lab)))
            else:
                lab = lab[:max_len]
            padded_labels.append(lab)
        batch_input["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch_input


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to JSONL dataset.")
    parser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--out_dir", type=str, default="./lora-llama3-mapping")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--val_ratio", type=float, default=0.02, help="Fraction for validation split.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Load dataset
    raw = load_dataset("json", data_files=args.data_path, split="train")

    # Basic filtering (keep only rows with both fields present)
    raw = raw.filter(lambda r: bool(r.get("input_text")) and bool(r.get("target_text")))

    # Train/val split
    val_size = max(1, int(len(raw) * args.val_ratio))
    ds = raw.train_test_split(test_size=val_size, seed=args.seed)
    train_ds, val_ds = ds["train"], ds["test"]

    # Tokenizer & model (4-bit + LoRA)
    print("Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading 4-bit base model (bitsandbytes)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype="auto",
    )

    print("Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)

    print("Attaching LoRA adapters...")
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    )
    model = get_peft_model(model, lora_cfg)

    # Collator masks non-assistant tokens
    collator = SupervisedDataCollator(tokenizer=tokenizer, max_len=args.max_len)

    # fp16/bf16 depending on HW
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        logging_steps=20,
        save_strategy="epoch",
        evaluation_strategy="steps",
        eval_steps=200,
        report_to="none",
        bf16=use_bf16,
        fp16=(not use_bf16),
        lr_scheduler_type="cosine",
    )

    # Map datasets to expected columns for the collator (no preprocessing; collator builds inputs)
    def keep_cols(ex):
        return {
            "input_text": ex["input_text"],
            "target_text": ex["target_text"],
        }

    train_ds = train_ds.map(keep_cols, remove_columns=[c for c in train_ds.column_names if c not in ["input_text","target_text"]])
    val_ds = val_ds.map(keep_cols, remove_columns=[c for c in val_ds.column_names if c not in ["input_text","target_text"]])

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print("Saving adapter & tokenizer...")
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    # Quick sanity eval loss on val
    metrics = trainer.evaluate()
    print("Eval metrics:", metrics)

    print("Done. Adapter dir:", args.out_dir)
    print("To use for inference, load base model + PEFT adapter, or merge weights with PEFT utilities if needed.")


if __name__ == "__main__":
    main()
