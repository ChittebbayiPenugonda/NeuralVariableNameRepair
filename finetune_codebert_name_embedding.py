"""
Finetune CodeBERT with an embedding-based loss for Neural Variable Name Repair.

Goal:
  For each (code snippet, variable name) pair, learn embeddings such that
  the code embedding is close to the gold variable-name embedding.

This script:
  - Uses DataPipeline to read your JSONL examples.
  - Expands each sample into (input_text, variable_name) pairs.
  - Uses microsoft/codebert-base as a shared encoder.
  - Adds small projection heads for code and name.
  - Trains with an MSE loss on embeddings (you can swap to cosine if desired).
  - Freezes most of CodeBERT and only tunes the "last layer" + projections.

You can later:
  - Use this as a dual-encoder for reranking LLM-generated variable names.
  - Or as an auxiliary semantic model.

Run on PACE (2 GPUs) e.g.:
  CUDA_VISIBLE_DEVICES=0,1 python finetune_codebert_name_embedding.py \\
      --data_file example_output.jsonl \\
      --output_dir ckpts/codebert_name_embed \\
      --epochs 3 --batch_size 16
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

from data_pipeline import DataPipeline


# ------------------------------
# Dataset: (code, variable_name) pairs
# ------------------------------

class VariableNamePairDataset(Dataset):
    """
    Expands each JSONL example into multiple (code, variable_name) pairs.

    For each entry:
      {
        "input_text": "... code with <ID_1>, <ID_2> ...",
        "target_text": "{\"<ID_1>\": \"count\", \"<ID_2>\": \"buffer_len\"}"
      }

    we create:
      (code, "count"), (code, "buffer_len")
    """

    def __init__(self, jsonl_file: str):
        self.examples: List[Dict[str, str]] = []

        pipeline = DataPipeline(jsonl_file)
        pipeline.load_data()

        for item in pipeline.data:
            code = item["input_text"]
            target_str = item["target_text"]
            try:
                mapping = json.loads(target_str)
            except json.JSONDecodeError:
                # Skip malformed targets
                continue

            for placeholder, name in mapping.items():
                # Skip weird or empty names
                if not isinstance(name, str) or not name.strip():
                    continue
                self.examples.append(
                    {"code": code, "name": name.strip(), "placeholder": placeholder}
                )

        print(f"Built dataset with {len(self.examples)} (code, name) pairs")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.examples[idx]


# ------------------------------
# Model: shared CodeBERT encoder + projection heads
# ------------------------------

class CodeToNameEmbeddingModel(nn.Module):
    """
    Shared CodeBERT encoder for both code and name text, with small
    projection heads to map to a final embedding space.

    We freeze most of the encoder and allow only the last transformer
    block + the projection layers to train, approximating "last-layer"
    fine-tuning.
    """

    def __init__(self, model_name: str = "microsoft/codebert-base"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        # Simple linear projections for code and names
        self.code_proj = nn.Linear(hidden_size, hidden_size)
        self.name_proj = nn.Linear(hidden_size, hidden_size)

        # Freeze everything by default
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Unfreeze the LAST transformer layer for a bit of adaptability
        if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
            print("Unfreezing last transformer layer of CodeBERT...")
            for param in self.encoder.encoder.layer[-1].parameters():
                param.requires_grad = True
        else:
            print("WARNING: encoder structure unexpected, leaving all layers frozen.")

        # Projections should be trainable
        for param in self.code_proj.parameters():
            param.requires_grad = True
        for param in self.name_proj.parameters():
            param.requires_grad = True

    def encode(self, input_ids, attention_mask):
        """
        Encode a batch of sequences using the [CLS] embedding.
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # For CodeBERT (RoBERTa-like), we use the first token as the sentence embedding
        cls_emb = outputs.last_hidden_state[:, 0, :]  # (batch, hidden)
        return cls_emb

    def forward(self, code_inputs, name_inputs):
        """
        code_inputs: dict with 'input_ids' and 'attention_mask' for code
        name_inputs: dict with 'input_ids' and 'attention_mask' for names

        Returns:
          code_emb: projected code embeddings
          name_emb: projected name embeddings
        """
        code_cls = self.encode(**code_inputs)
        name_cls = self.encode(**name_inputs)

        code_emb = self.code_proj(code_cls)
        name_emb = self.name_proj(name_cls)

        return code_emb, name_emb


# ------------------------------
# Collate function
# ------------------------------

def make_collate_fn(tokenizer, max_length_code: int = 256, max_length_name: int = 16):
    def collate(batch: List[Dict[str, str]]):
        codes = [b["code"] for b in batch]
        names = [b["name"] for b in batch]

        code_enc = tokenizer(
            codes,
            padding=True,
            truncation=True,
            max_length=max_length_code,
            return_tensors="pt",
        )
        name_enc = tokenizer(
            names,
            padding=True,
            truncation=True,
            max_length=max_length_name,
            return_tensors="pt",
        )
        return code_enc, name_enc

    return collate


# ------------------------------
# Training loop
# ------------------------------

def train(
    data_file: str,
    model_name: str,
    output_dir: str,
    batch_size: int = 16,
    epochs: int = 3,
    lr: float = 1e-4,
    warmup_steps: int = 100,
    max_length_code: int = 256,
    max_length_name: int = 16,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Dataset + DataLoader
    dataset = VariableNamePairDataset(data_file)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    collate_fn = make_collate_fn(tokenizer, max_length_code, max_length_name)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Model
    model = CodeToNameEmbeddingModel(model_name=model_name)
    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel over {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model.to(device)

    # Only parameters with requires_grad=True are optimized
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(
        f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}"
    )

    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    total_steps = epochs * len(dataloader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # MSE on embeddings (you can swap to cosine loss if you prefer)
    criterion = nn.MSELoss()

    global_step = 0
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (code_enc, name_enc) in enumerate(dataloader):
            code_enc = {k: v.to(device) for k, v in code_enc.items()}
            name_enc = {k: v.to(device) for k, v in name_enc.items()}

            optimizer.zero_grad()

            code_emb, name_emb = model(code_inputs=code_enc, name_inputs=name_enc)

            loss = criterion(code_emb, name_emb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            global_step += 1

            if (batch_idx + 1) % 50 == 0:
                avg_loss = running_loss / 50
                print(
                    f"Epoch [{epoch+1}/{epochs}] "
                    f"Step [{batch_idx+1}/{len(dataloader)}] "
                    f"Loss: {avg_loss:.4f}"
                )
                running_loss = 0.0

        # Save checkpoint each epoch
        ckpt_dir = output_path / f"epoch_{epoch+1}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(model, nn.DataParallel):
            model.module.encoder.save_pretrained(ckpt_dir / "encoder")
        else:
            model.encoder.save_pretrained(ckpt_dir / "encoder")
        tokenizer.save_pretrained(ckpt_dir / "encoder")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
            },
            ckpt_dir / "model.pt",
        )
        print(f"Saved checkpoint to {ckpt_dir}")

    print("Training complete!")


# ------------------------------
# CLI
# ------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Finetune CodeBERT embedding model for variable name semantics"
    )
    parser.add_argument("--data_file", type=str, required=True,
                        help="Path to JSONL file (same format as example_output.jsonl)")
    parser.add_argument("--model_name", type=str, default="microsoft/codebert-base",
                        help="Base model name")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save checkpoints")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_length_code", type=int, default=256)
    parser.add_argument("--max_length_name", type=int, default=16)

    args = parser.parse_args()

    train(
        data_file=args.data_file,
        model_name=args.model_name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        max_length_code=args.max_length_code,
        max_length_name=args.max_length_name,
    )


if __name__ == "__main__":
    main()
