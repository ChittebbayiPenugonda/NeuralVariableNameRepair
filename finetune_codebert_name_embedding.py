#!/usr/bin/env python3
"""
Finetune CodeBERT as a dual encoder for (code, variable name) pairs.

Goal (H3 support):
  - Learn embeddings such that the masked code snippet embedding is close to
    the embedding of the correct variable name.
  - Use an InfoNCE-style contrastive loss over in-batch negatives.

This script supersedes the earlier MSE-based version we discussed.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel

from data_pipeline import DataPipeline


# -------------------------------
# Dataset: (code, variable_name) pairs
# -------------------------------

class CodeNamePairDataset(Dataset):
    """
    Builds (code, variable_name) pairs from the JSONL data.

    Each data item typically has:
      - input_text: masked C++ function
      - target_text: JSON string mapping "<ID_i>" -> "variableName"

    We iterate over each mapping entry and create a separate pair.
    """

    def __init__(self, data_file: str, min_name_len: int = 1):
        pipeline = DataPipeline(data_file)
        pipeline.load_data()

        self.pairs: List[Tuple[str, str]] = []

        for item in pipeline.data:
            code = item.get("input_text", "")
            target = item.get("target_text", "")
            if not code or not target:
                continue

            try:
                mapping = json.loads(target)
                if not isinstance(mapping, dict):
                    continue
            except Exception:
                # If parsing fails, skip this example
                continue

            for name in mapping.values():
                if not isinstance(name, str):
                    continue
                name = name.strip()
                if len(name) >= min_name_len:
                    self.pairs.append((code, name))

        if not self.pairs:
            raise ValueError("No (code, name) pairs constructed from data file.")

        print(f"Built {len(self.pairs)} (code, name) pairs from {data_file}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return self.pairs[idx]


# -------------------------------
# Model: CodeBERT dual encoder
# -------------------------------

class CodeToNameEmbeddingModel(nn.Module):
    """
    Dual encoder:
      - Shared CodeBERT backbone encodes both code and name text.
      - Separate projection heads for code and name.
      - InfoNCE contrastive loss over in-batch negatives.

    We freeze all CodeBERT layers except the last transformer block.
    """

    def __init__(self, model_name: str = "microsoft/codebert-base"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.code_proj = nn.Linear(hidden_size, hidden_size)
        self.name_proj = nn.Linear(hidden_size, hidden_size)

        # Optional: learned temperature; we clamp its exp for stability
        self.log_temperature = nn.Parameter(torch.tensor(-2.0))  # exp(-2) ≈ 0.135

        # Freeze all encoder params
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Unfreeze last transformer layer (if present)
        encoder_attr = getattr(self.encoder, "encoder", None)
        if encoder_attr is not None and hasattr(encoder_attr, "layer"):
            last_layer = encoder_attr.layer[-1]
            for p in last_layer.parameters():
                p.requires_grad = True
            print("Unfroze last encoder layer for fine-tuning.")
        else:
            print("WARNING: Could not find encoder.layer[-1] to unfreeze; training projections only.")

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of tokenized inputs and return [CLS] embeddings.
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # CodeBERT uses [CLS] at position 0
        cls_emb = outputs.last_hidden_state[:, 0, :]
        return cls_emb

    def forward(
        self,
        code_input_ids: torch.Tensor,
        code_attention_mask: torch.Tensor,
        name_input_ids: torch.Tensor,
        name_attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        code_cls = self.encode(code_input_ids, code_attention_mask)
        name_cls = self.encode(name_input_ids, name_attention_mask)

        code_emb = F.normalize(self.code_proj(code_cls), p=2, dim=-1)
        name_emb = F.normalize(self.name_proj(name_cls), p=2, dim=-1)
        return code_emb, name_emb

    def contrastive_loss(self, code_emb: torch.Tensor, name_emb: torch.Tensor) -> torch.Tensor:
        """
        Symmetric InfoNCE loss over in-batch negatives.

        For batch size B:
          - similarity matrix S_ij = code_emb[i] · name_emb[j] / T
          - positives are on the diagonal.
        """
        batch_size = code_emb.size(0)
        temperature = torch.clamp(self.log_temperature.exp(), 1e-4, 1.0)

        # (B, B) similarity matrix
        logits = code_emb @ name_emb.t() / temperature

        labels = torch.arange(batch_size, device=code_emb.device)

        loss_i = F.cross_entropy(logits, labels)
        loss_j = F.cross_entropy(logits.t(), labels)

        return (loss_i + loss_j) * 0.5


# -------------------------------
# Collator
# -------------------------------

class CodeNameCollator:
    """
    Tokenize code and name text into model inputs.
    """

    def __init__(self, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Tuple[str, str]]) -> Dict[str, torch.Tensor]:
        codes, names = zip(*batch)

        code_enc = self.tokenizer(
            list(codes),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        name_enc = self.tokenizer(
            list(names),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "code_input_ids": code_enc["input_ids"],
            "code_attention_mask": code_enc["attention_mask"],
            "name_input_ids": name_enc["input_ids"],
            "name_attention_mask": name_enc["attention_mask"],
        }


# -------------------------------
# Training loop
# -------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune CodeBERT dual encoder for (code, variable name) embeddings"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="JSONL input (same as example_output.jsonl)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save model checkpoint",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/codebert-base",
        help="Backbone encoder model name",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Max token length for code/name",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=50,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Dataset + tokenizer
    dataset = CodeNamePairDataset(args.data_file)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    collator = CodeNameCollator(tokenizer, max_length=args.max_length)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    # Model
    model = CodeToNameEmbeddingModel(args.model_name)
    model.to(device)

    # Multi-GPU (optional)
    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel across {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)

    # Optimizer & scheduler
    # Count trainable params for sanity
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {total_trainable}")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    num_training_steps = args.num_epochs * len(dataloader)
    warmup_steps = min(args.warmup_steps, num_training_steps // 2)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        # Linear decay after warmup
        return max(0.0, float(num_training_steps - step) / max(1, num_training_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    global_step = 0
    model.train()
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}"):
            code_input_ids = batch["code_input_ids"].to(device)
            code_attention_mask = batch["code_attention_mask"].to(device)
            name_input_ids = batch["name_input_ids"].to(device)
            name_attention_mask = batch["name_attention_mask"].to(device)

            optimizer.zero_grad()

            code_emb, name_emb = model(
                code_input_ids=code_input_ids,
                code_attention_mask=code_attention_mask,
                name_input_ids=name_input_ids,
                name_attention_mask=name_attention_mask,
            )

            # DataParallel wraps the module; contrastive_loss lives on the real module
            if isinstance(model, nn.DataParallel):
                loss = model.module.contrastive_loss(code_emb, name_emb)
            else:
                loss = model.contrastive_loss(code_emb, name_emb)

            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % args.log_interval == 0:
                avg_loss = epoch_loss / max(1, global_step)
                current_lr = scheduler.get_last_lr()[0]
                print(f"Step {global_step} | avg_loss={avg_loss:.4f} | lr={current_lr:.6e}")

        avg_epoch_loss = epoch_loss / max(1, len(dataloader))
        print(f"Epoch {epoch+1} complete | avg_epoch_loss={avg_epoch_loss:.4f}")

    # Save checkpoint
    save_path = Path(args.output_dir) / "dual_encoder_codebert.pt"

    # Get underlying module if DataParallel
    model_to_save = model.module if isinstance(model, nn.DataParallel) else model

    torch.save(
        {
            "model_state_dict": model_to_save.state_dict(),
            "model_name": args.model_name,
            "args": vars(args),
        },
        save_path,
    )

    print(f"Saved dual encoder checkpoint to {save_path}")


if __name__ == "__main__":
    main()
