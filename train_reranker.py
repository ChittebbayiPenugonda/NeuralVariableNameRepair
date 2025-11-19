"""
Train the Dual-Encoder Reranker on Variable Name Repair Data

This script:
1. Loads training data (masked code + gold mappings)
2. Generates negative samples (wrong names from generator or corpus)
3. Trains the reranker with contrastive InfoNCE loss
4. Saves the trained model for inference

Usage:
    python train_reranker.py \
        --data_path ./training_data.jsonl \
        --generator_predictions ./generator_top_k.jsonl \
        --output_dir ./reranker_checkpoint \
        --epochs 5 --batch_size 32
"""

import os
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from reranker import (
    DualEncoderReranker,
    ContextExtractor,
    SubtokenSplitter,
)


class RerankerDataset(Dataset):
    """
    Dataset for training reranker with contrastive learning.
    Each sample is a tuple: (context, positive_name, negative_name)
    """
    
    def __init__(
        self,
        data: List[Tuple[str, str, str]],
    ):
        """
        Args:
            data: List of (context, gold_name, negative_name) tuples
        """
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def load_training_data(
    data_path: str,
    context_window: int = 100,
) -> List[Dict]:
    """
    Load JSONL training data.
    
    Expected format per line:
    {
        "input_text": "<code with placeholders>",
        "target_text": '{"<ID_1>": "name1", "<ID_2>": "name2"}'
    }
    
    Returns:
        List of parsed data dictionaries
    """
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                sample = json.loads(line)
                # Parse target JSON
                target_dict = json.loads(sample['target_text'])
                sample['target_dict'] = target_dict
                data.append(sample)
    
    print(f"Loaded {len(data)} training samples from {data_path}")
    return data


def load_generator_predictions(
    predictions_path: str,
) -> Dict[int, Dict[str, List[str]]]:
    """
    Load generator top-k predictions for negative sampling.
    
    Expected format per line:
    {
        "index": 0,
        "input_text": "<code>",
        "target_text": '{"<ID_1>": "name"}',
        "top_k_predictions": {
            "<ID_1>": ["pred1", "pred2", "pred3", ...]
        }
    }
    
    Returns:
        Dict mapping sample index -> placeholder -> list of predictions
    """
    predictions = {}
    
    if not os.path.exists(predictions_path):
        print(f"Warning: Generator predictions file not found: {predictions_path}")
        print("Will use random negatives from other samples instead.")
        return {}
    
    with open(predictions_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                pred = json.loads(line)
                idx = pred['index']
                predictions[idx] = pred.get('top_k_predictions', {})
    
    print(f"Loaded generator predictions for {len(predictions)} samples")
    return predictions


def create_contrastive_samples(
    training_data: List[Dict],
    generator_predictions: Dict[int, Dict[str, List[str]]],
    context_window: int = 100,
    num_negatives_per_positive: int = 1,
) -> List[Tuple[str, str, str]]:
    """
    Create (context, positive_name, negative_name) tuples for contrastive training.
    
    Negative sampling strategies:
    1. Use incorrect predictions from generator (if available)
    2. Use names from other placeholders in the same function
    3. Use names from other functions in the dataset
    
    Args:
        training_data: List of training samples
        generator_predictions: Generator top-k predictions for hard negatives
        context_window: Size of context window
        num_negatives_per_positive: Number of negative samples per positive
        
    Returns:
        List of (context, positive_name, negative_name) tuples
    """
    extractor = ContextExtractor(window_size=context_window)
    contrastive_samples = []
    
    # Collect all names for random negative sampling
    all_names = []
    for sample in training_data:
        all_names.extend(sample['target_dict'].values())
    all_names = list(set(all_names))
    
    print("Creating contrastive training samples...")
    for idx, sample in enumerate(tqdm(training_data)):
        code = sample['input_text']
        gold_mapping = sample['target_dict']
        
        for placeholder, gold_name in gold_mapping.items():
            # Extract context around this placeholder
            context = extractor.extract_context(code, placeholder)
            
            # Get negative samples
            negatives = set()
            
            # Strategy 1: Hard negatives from generator predictions
            if idx in generator_predictions and placeholder in generator_predictions[idx]:
                gen_preds = generator_predictions[idx][placeholder]
                for pred in gen_preds:
                    if pred != gold_name:  # Only incorrect predictions
                        negatives.add(pred)
                        if len(negatives) >= num_negatives_per_positive:
                            break
            
            # Strategy 2: Names from other placeholders in same function
            if len(negatives) < num_negatives_per_positive:
                other_names = [v for k, v in gold_mapping.items() if k != placeholder]
                negatives.update(other_names[:num_negatives_per_positive - len(negatives)])
            
            # Strategy 3: Random names from corpus
            if len(negatives) < num_negatives_per_positive:
                candidates = [n for n in all_names if n != gold_name and n not in negatives]
                if candidates:
                    num_needed = num_negatives_per_positive - len(negatives)
                    random_negs = random.sample(candidates, min(num_needed, len(candidates)))
                    negatives.update(random_negs)
            
            # Create training samples
            for neg_name in list(negatives)[:num_negatives_per_positive]:
                contrastive_samples.append((context, gold_name, neg_name))
    
    print(f"Created {len(contrastive_samples)} contrastive training samples")
    return contrastive_samples


def train_reranker(
    reranker: DualEncoderReranker,
    train_samples: List[Tuple[str, str, str]],
    val_samples: List[Tuple[str, str, str]],
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = "./reranker_checkpoint",
) -> DualEncoderReranker:
    """
    Train reranker with InfoNCE contrastive loss.
    
    Args:
        reranker: Model to train
        train_samples: Training data (context, positive, negative)
        val_samples: Validation data
        epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device to train on
        output_dir: Directory to save checkpoints
        
    Returns:
        Trained reranker
    """
    import torch.nn.functional as F
    
    reranker = reranker.to(device)
    optimizer = torch.optim.AdamW(reranker.parameters(), lr=lr, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs * (len(train_samples) // batch_size),
        eta_min=lr * 0.1,
    )
    
    def infonce_loss(context_embeds, pos_embeds, neg_embeds, temperature):
        """InfoNCE contrastive loss."""
        # Positive similarity
        pos_sim = (context_embeds * pos_embeds).sum(dim=-1) / temperature
        
        # Negative similarity
        neg_sim = (context_embeds * neg_embeds).sum(dim=-1) / temperature
        
        # Stack and compute cross-entropy
        logits = torch.stack([pos_sim, neg_sim], dim=-1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        
        loss = F.cross_entropy(logits, labels)
        
        # Accuracy for monitoring
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()
        
        return loss, acc
    
    def evaluate(val_samples, batch_size=64):
        """Evaluate on validation set."""
        reranker.eval()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(val_samples), batch_size):
                batch = val_samples[i:i+batch_size]
                contexts, pos_names, neg_names = zip(*batch)
                
                context_embeds = reranker.encode_context(list(contexts))
                pos_embeds = reranker.encode_names(list(pos_names))
                neg_embeds = reranker.encode_names(list(neg_names))
                
                loss, acc = infonce_loss(
                    context_embeds,
                    pos_embeds,
                    neg_embeds,
                    reranker.temperature.clamp(min=1e-3),
                )
                
                total_loss += loss.item()
                total_acc += acc.item()
                num_batches += 1
        
        return total_loss / num_batches, total_acc / num_batches
    
    print(f"\nTraining reranker on {device}")
    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Batch size: {batch_size}, Epochs: {epochs}, LR: {lr}\n")
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        reranker.train()
        random.shuffle(train_samples)
        
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        
        pbar = tqdm(range(0, len(train_samples), batch_size), desc=f"Epoch {epoch+1}/{epochs}")
        for i in pbar:
            batch = train_samples[i:i+batch_size]
            if len(batch) == 0:
                continue
            
            contexts, pos_names, neg_names = zip(*batch)
            
            # Encode
            context_embeds = reranker.encode_context(list(contexts))
            pos_embeds = reranker.encode_names(list(pos_names))
            neg_embeds = reranker.encode_names(list(neg_names))
            
            # Compute loss
            loss, acc = infonce_loss(
                context_embeds,
                pos_embeds,
                neg_embeds,
                reranker.temperature.clamp(min=1e-3),
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reranker.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc.item():.3f}',
                'temp': f'{reranker.temperature.item():.4f}',
            })
        
        # Epoch summary
        avg_train_loss = epoch_loss / num_batches
        avg_train_acc = epoch_acc / num_batches
        
        # Validation
        val_loss, val_acc = evaluate(val_samples, batch_size=64)
        
        print(f"\nEpoch {epoch+1}/{epochs} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.3f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}")
        print(f"  Temperature: {reranker.temperature.item():.4f}")
        
        # Save checkpoint
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = output_path / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': reranker.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
        }, checkpoint_path)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = output_path / "best_model.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': reranker.state_dict(),
                'val_loss': val_loss,
            }, best_path)
            print(f"  → Saved best model (val_loss: {val_loss:.4f})")
        
        print()
    
    print("Training complete!")
    return reranker


def main():
    parser = argparse.ArgumentParser(description='Train dual-encoder reranker')
    
    # Data paths
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to training JSONL file')
    parser.add_argument('--generator_predictions', type=str, default=None,
                        help='Path to generator top-k predictions (for hard negatives)')
    parser.add_argument('--output_dir', type=str, default='./reranker_checkpoint',
                        help='Directory to save model checkpoints')
    
    # Model config
    parser.add_argument('--base_model', type=str, default='microsoft/codebert-base',
                        help='Base encoder model (CodeBERT, GraphCodeBERT, etc.)')
    parser.add_argument('--hidden_dim', type=int, default=768,
                        help='Hidden dimension of encoders')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability')
    parser.add_argument('--freeze_base', action='store_true',
                        help='Freeze base encoder weights (only train projection heads)')
    
    # Training config
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation split ratio')
    parser.add_argument('--context_window', type=int, default=100,
                        help='Context window size around placeholders')
    parser.add_argument('--negatives_per_positive', type=int, default=1,
                        help='Number of negative samples per positive')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    training_data = load_training_data(args.data_path, args.context_window)
    
    # Load generator predictions for hard negatives (optional)
    generator_preds = {}
    if args.generator_predictions:
        generator_preds = load_generator_predictions(args.generator_predictions)
    
    # Create contrastive samples
    print("\n" + "="*80)
    print("CREATING CONTRASTIVE SAMPLES")
    print("="*80)
    all_samples = create_contrastive_samples(
        training_data,
        generator_preds,
        context_window=args.context_window,
        num_negatives_per_positive=args.negatives_per_positive,
    )
    
    # Train/val split
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * (1 - args.val_ratio))
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    print(f"\nTrain samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    
    # Initialize reranker
    print("\n" + "="*80)
    print("INITIALIZING RERANKER")
    print("="*80)
    print(f"Base model: {args.base_model}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Dropout: {args.dropout}")
    print(f"Freeze base: {args.freeze_base}")
    
    reranker = DualEncoderReranker(
        model_name=args.base_model,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        freeze_base=args.freeze_base,
    )
    
    # Train
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    trained_reranker = train_reranker(
        reranker,
        train_samples,
        val_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
    )
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"Checkpoints saved to: {args.output_dir}")
    print(f"Load best model from: {args.output_dir}/best_model.pt")


if __name__ == "__main__":
    main()

