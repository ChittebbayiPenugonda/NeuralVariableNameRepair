"""
Finetune Llama 3.1 8B Instruct with LoRA on the Neural Variable Name Repair task.

This script:
  - Uses DataPipeline to load (input_text, target_text) pairs from a JSONL file.
  - Formats each example using the same chat template as run_inference.py.
  - Trains a LoRA adapter on top of the base model so that it maps masked code
    → JSON mapping of <ID_i> → variable names.
  - Supports linear warmup (H1) and configurable dropout (H1) on attention/MLP.
  - Produces a LoRA checkpoint directory you can later load for inference.

Typical usage on PACE (2 GPUs):
  HF_TOKEN=... CUDA_VISIBLE_DEVICES=0,1 \\
  python finetune_llama_lora.py \\
      --data_file example_output.jsonl \\
      --output_dir ckpts/llama_lora \\
      --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \\
      --num_train_epochs 2 \\
      --per_device_train_batch_size 1 \\
      --gradient_accumulation_steps 8 \\
      --warmup_steps 500 \\
      --learning_rate 1e-4
"""

import os
import json
import math
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, get_peft_model

from data_pipeline import DataPipeline


# -------------------------------
# Prompt helpers (copied from run_inference.py)
# -------------------------------

def load_prompt_template(prompt_file: str = "prompt.txt") -> str:
    """Load the system prompt template from file."""
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read().strip()


def format_chat_prompt(system_prompt: str, user_input: str) -> str:
    """
    Format prompt for Llama 3.1 Instruct using chat template format.
    Same as run_inference.format_chat_prompt.
    """
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


# -------------------------------
# Dataset
# -------------------------------

class RepairDataset(Dataset):
    """
    Dataset of (prompt, target_json) pairs for variable name repair.

    For each entry:
      input_text: masked code
      target_text: JSON mapping string
    we build a training example where the model sees:

      [prompt(system, code)]  →  [target_text]
    """

    def __init__(
        self,
        data_file: str,
        tokenizer: AutoTokenizer,
        prompt_template: str,
        max_length: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        pipeline = DataPipeline(data_file)
        pipeline.load_data()
        input_texts, target_texts = pipeline.get_separate_arrays()

        self.examples: List[Tuple[str, str]] = []
        for code, target in zip(input_texts, target_texts):
            # Skip bad targets
            if not isinstance(target, str) or not target.strip():
                continue
            self.examples.append((code, target.strip()))

        self.system_prompt = prompt_template

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        code, target = self.examples[idx]
        # Build chat-style prompt
        prompt_str = format_chat_prompt(self.system_prompt, code)

        # Full text = prompt + target JSON (no extra text)
        full_text = prompt_str + target

        # Tokenize full text
        enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]

        # To train only on the assistant part, we mask out prompt tokens in labels
        with self.tokenizer.as_target_tokenizer():
            prompt_enc = self.tokenizer(
                prompt_str,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                add_special_tokens=False,
                return_tensors="pt",
            )

        prompt_len = prompt_enc["input_ids"].shape[1]
        labels = input_ids.clone()

        # Mask out prompt tokens
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# -------------------------------
# Main training logic
# -------------------------------

def build_model_and_tokenizer(
    model_name: str,
    attn_dropout: float,
    ffn_dropout: float,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load base model + tokenizer and apply dropout settings."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Some safety tweaks for training
    model.config.use_cache = False

    # Apply dropout settings (H1)
    if hasattr(model.config, "attention_dropout"):
        model.config.attention_dropout = attn_dropout
    if hasattr(model.config, "hidden_dropout"):
        model.config.hidden_dropout = ffn_dropout
    if hasattr(model.config, "resid_pdrop"):
        model.config.resid_pdrop = ffn_dropout
    if hasattr(model.config, "embd_pdrop"):
        model.config.embd_pdrop = ffn_dropout

    return model, tokenizer


def apply_lora(model) -> AutoModelForCausalLM:
    """Wrap the base model with LoRA adapters (H2)."""
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune Llama 3.1 8B with LoRA for Neural Variable Name Repair"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to JSONL file (same format as example_output.jsonl)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save LoRA adapter and checkpoints",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Base HF model name",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="prompt.txt",
        help="System prompt file (same as run_inference.py)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Max sequence length for training examples",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Linear warmup steps (H1)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--attn_dropout",
        type=float,
        default=0.1,
        help="Attention dropout probability (H1)",
    )
    parser.add_argument(
        "--ffn_dropout",
        type=float,
        default=0.1,
        help="MLP / residual dropout probability (H1)",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 training if available",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load prompt
    system_prompt = load_prompt_template(args.prompt_file)

    # Build model + tokenizer
    model, tokenizer = build_model_and_tokenizer(
        args.model_name,
        attn_dropout=args.attn_dropout,
        ffn_dropout=args.ffn_dropout,
    )

    # Apply LoRA
    model = apply_lora(model)

    # Build dataset
    train_dataset = RepairDataset(
        data_file=args.data_file,
        tokenizer=tokenizer,
        prompt_template=system_prompt,
        max_length=args.max_length,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=args.bf16,
        optim="adamw_torch",
        lr_scheduler_type="linear",
        report_to="none",
        run_name="llama_lora_neural_var_name_repair",
        gradient_checkpointing=True,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save only the LoRA adapter (smaller & what we actually need)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA-tuned model to {args.output_dir}")


if __name__ == "__main__":
    main()
