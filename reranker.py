"""
Dual-Encoder Reranker for Variable Name Candidates

This module implements a reranking system that:
1. Takes top-k candidate names per placeholder from the generator
2. Scores each candidate using a dual-encoder architecture
3. Selects the best candidate based on context fit

Architecture:
- Code Context Encoder: Embeds the local context around each placeholder
- Name Encoder: Embeds candidate identifier names (subtoken-aware)
- Scoring: Cosine similarity with learned temperature + penalties
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from transformers import AutoModel, AutoTokenizer


@dataclass
class RerankCandidate:
    """Represents a candidate name for a single placeholder."""
    placeholder: str  # e.g., "<ID_1>"
    candidate_name: str  # e.g., "fileName"
    context: str  # Code snippet around the placeholder
    score: float = 0.0  # Reranker score


class SubtokenSplitter:
    """
    Splits identifiers into subtokens using camelCase and snake_case conventions.
    Example: "fileName" → ["file", "name"]
             "file_name" → ["file", "name"]
    """
    
    @staticmethod
    def split(identifier: str) -> List[str]:
        """Split identifier into subtokens."""
        # Handle snake_case
        identifier = identifier.replace('_', ' ')
        
        # Handle camelCase: insert space before uppercase letters
        identifier = re.sub(r'([a-z])([A-Z])', r'\1 \2', identifier)
        
        # Split and filter empty strings
        subtokens = [t.lower() for t in identifier.split() if t]
        
        return subtokens if subtokens else [identifier.lower()]
    
    @staticmethod
    def join_subtokens(subtokens: List[str]) -> str:
        """Join subtokens into a single string for embedding."""
        return " ".join(subtokens)


class ContextExtractor:
    """
    Extracts local context around a placeholder in code.
    Includes surrounding tokens, type annotations, and operations.
    """
    
    def __init__(self, window_size: int = 50):
        """
        Args:
            window_size: Number of characters to include before/after placeholder
        """
        self.window_size = window_size
    
    def extract_context(self, code: str, placeholder: str) -> str:
        """
        Extract local context around the first occurrence of placeholder.
        
        Args:
            code: Full function code
            placeholder: The placeholder to find (e.g., "<ID_1>")
            
        Returns:
            Context string with placeholder marked
        """
        idx = code.find(placeholder)
        if idx == -1:
            return code[:self.window_size * 2]  # Fallback
        
        start = max(0, idx - self.window_size)
        end = min(len(code), idx + len(placeholder) + self.window_size)
        
        context = code[start:end]
        return context
    
    def extract_all_contexts(self, code: str, placeholders: List[str]) -> Dict[str, str]:
        """Extract contexts for all placeholders in code."""
        contexts = {}
        for ph in placeholders:
            contexts[ph] = self.extract_context(code, ph)
        return contexts


class DualEncoderReranker(nn.Module):
    """
    Dual-encoder architecture for reranking variable name candidates.
    
    Architecture:
    - Context Encoder: Encodes code context around placeholder (CodeBERT-based)
    - Name Encoder: Encodes candidate identifier names (subtoken-aware)
    - Scoring: Cosine similarity with learned temperature parameter
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        hidden_dim: int = 768,
        dropout: float = 0.1,
        freeze_base: bool = False,
    ):
        """
        Args:
            model_name: HuggingFace model for encoders (CodeBERT, GraphCodeBERT, etc.)
            hidden_dim: Dimension of encoder outputs
            dropout: Dropout probability
            freeze_base: Whether to freeze pretrained weights
        """
        super().__init__()
        
        # Load shared base model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model = AutoModel.from_pretrained(model_name)
        
        # Context encoder (for code snippets)
        self.context_encoder = base_model
        if freeze_base:
            for param in self.context_encoder.parameters():
                param.requires_grad = False
        
        # Name encoder (shares same architecture, different parameters)
        self.name_encoder = AutoModel.from_pretrained(model_name)
        if freeze_base:
            for param in self.name_encoder.parameters():
                param.requires_grad = False
        
        # Projection heads to normalize representations
        self.context_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.name_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Learnable temperature for similarity scaling
        self.temperature = nn.Parameter(torch.tensor(0.07))
        
        # Length bias (slight preference for concise names)
        self.length_weight = nn.Parameter(torch.tensor(-0.05))
        
        self.subtoken_splitter = SubtokenSplitter()
    
    def encode_context(self, contexts: List[str]) -> torch.Tensor:
        """
        Encode code contexts into embeddings.
        
        Args:
            contexts: List of code context strings
            
        Returns:
            Tensor of shape (batch_size, hidden_dim)
        """
        # Tokenize contexts
        encoded = self.tokenizer(
            contexts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        
        # Move to same device as model
        device = next(self.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # Get [CLS] token representation
        outputs = self.context_encoder(**encoded)
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Project to scoring space
        projected = self.context_proj(pooled)
        
        # L2 normalize
        normalized = F.normalize(projected, p=2, dim=-1)
        
        return normalized
    
    def encode_names(self, names: List[str]) -> torch.Tensor:
        """
        Encode candidate identifier names into embeddings.
        Names are split into subtokens before encoding.
        
        Args:
            names: List of candidate identifier names
            
        Returns:
            Tensor of shape (batch_size, hidden_dim)
        """
        # Split into subtokens
        subtoken_texts = [
            self.subtoken_splitter.join_subtokens(self.subtoken_splitter.split(name))
            for name in names
        ]
        
        # Tokenize
        encoded = self.tokenizer(
            subtoken_texts,
            padding=True,
            truncation=True,
            max_length=32,
            return_tensors="pt",
        )
        
        # Move to device
        device = next(self.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # Get [CLS] token representation
        outputs = self.name_encoder(**encoded)
        pooled = outputs.last_hidden_state[:, 0, :]
        
        # Project
        projected = self.name_proj(pooled)
        
        # L2 normalize
        normalized = F.normalize(projected, p=2, dim=-1)
        
        return normalized
    
    def compute_similarity_scores(
        self,
        context_embeds: torch.Tensor,
        name_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cosine similarity scores between contexts and names.
        
        Args:
            context_embeds: (batch_size, hidden_dim)
            name_embeds: (num_candidates, hidden_dim)
            
        Returns:
            Similarity matrix of shape (batch_size, num_candidates)
        """
        # Cosine similarity (already normalized, so just dot product)
        similarity = torch.matmul(context_embeds, name_embeds.T)
        
        # Scale by learned temperature
        scaled_similarity = similarity / self.temperature.clamp(min=1e-3)
        
        return scaled_similarity
    
    def forward(
        self,
        contexts: List[str],
        candidate_names: List[List[str]],
    ) -> torch.Tensor:
        """
        Forward pass: score all candidates for each context.
        
        Args:
            contexts: List of code contexts (one per placeholder)
            candidate_names: List of candidate lists (one list per placeholder)
            
        Returns:
            Scores tensor of shape (num_placeholders, max_k)
        """
        # Encode contexts
        context_embeds = self.encode_context(contexts)  # (num_placeholders, hidden_dim)
        
        # Flatten all candidates for batch encoding
        all_candidates = []
        candidate_counts = []
        for candidates in candidate_names:
            all_candidates.extend(candidates)
            candidate_counts.append(len(candidates))
        
        # Encode all candidates
        if not all_candidates:
            return torch.zeros((len(contexts), 0))
        
        name_embeds = self.encode_names(all_candidates)  # (total_candidates, hidden_dim)
        
        # Compute similarity for each placeholder's candidates
        scores_list = []
        cand_idx = 0
        for i, count in enumerate(candidate_counts):
            # Get this placeholder's context and candidates
            ctx_embed = context_embeds[i:i+1]  # (1, hidden_dim)
            cand_embeds = name_embeds[cand_idx:cand_idx+count]  # (count, hidden_dim)
            
            # Similarity scores
            sim_scores = torch.matmul(ctx_embed, cand_embeds.T) / self.temperature.clamp(min=1e-3)
            sim_scores = sim_scores.squeeze(0)  # (count,)
            
            # Length penalty (slight preference for shorter names)
            lengths = torch.tensor(
                [len(all_candidates[cand_idx + j]) for j in range(count)],
                device=sim_scores.device,
                dtype=sim_scores.dtype,
            )
            length_penalty = self.length_weight * lengths
            
            # Final scores
            final_scores = sim_scores + length_penalty
            
            scores_list.append(final_scores)
            cand_idx += count
        
        return scores_list


class RerankerPipeline:
    """
    End-to-end reranking pipeline that:
    1. Extracts contexts for each placeholder
    2. Scores all candidates
    3. Applies collision penalties
    4. Returns best candidates
    """
    
    def __init__(
        self,
        reranker_model: DualEncoderReranker,
        context_window: int = 100,
        collision_penalty: float = -5.0,
    ):
        """
        Args:
            reranker_model: Trained dual-encoder reranker
            context_window: Size of context window around placeholders
            collision_penalty: Score penalty for naming collisions
        """
        self.reranker = reranker_model
        self.context_extractor = ContextExtractor(window_size=context_window)
        self.collision_penalty = collision_penalty
    
    def rerank(
        self,
        code: str,
        candidates_dict: Dict[str, List[str]],
    ) -> Dict[str, str]:
        """
        Rerank candidates and return best name for each placeholder.
        
        Args:
            code: Full function code with placeholders
            candidates_dict: Dict mapping placeholder -> list of candidate names
                Example: {"<ID_1>": ["x", "count", "index"], "<ID_2>": ["sum", "total"]}
        
        Returns:
            Dict mapping placeholder -> best candidate name
        """
        if not candidates_dict:
            return {}
        
        placeholders = list(candidates_dict.keys())
        
        # Extract contexts
        contexts = [
            self.context_extractor.extract_context(code, ph)
            for ph in placeholders
        ]
        
        # Get candidate lists
        candidate_lists = [candidates_dict[ph] for ph in placeholders]
        
        # Score candidates
        self.reranker.eval()
        with torch.no_grad():
            scores_lists = self.reranker(contexts, candidate_lists)
        
        # Select best candidates with collision detection
        selected = {}
        used_names = set()
        
        # Sort placeholders by confidence (max score) to assign high-confidence names first
        placeholder_max_scores = [
            (ph, scores.max().item() if len(scores) > 0 else -float('inf'))
            for ph, scores in zip(placeholders, scores_lists)
        ]
        sorted_placeholders = sorted(placeholder_max_scores, key=lambda x: x[1], reverse=True)
        
        for ph, _ in sorted_placeholders:
            ph_idx = placeholders.index(ph)
            candidates = candidate_lists[ph_idx]
            scores = scores_lists[ph_idx]
            
            if len(candidates) == 0:
                selected[ph] = "unknown"
                continue
            
            # Apply collision penalties
            adjusted_scores = scores.clone()
            for i, cand in enumerate(candidates):
                if cand in used_names:
                    adjusted_scores[i] += self.collision_penalty
            
            # Select best
            best_idx = adjusted_scores.argmax().item()
            best_name = candidates[best_idx]
            
            selected[ph] = best_name
            used_names.add(best_name)
        
        return selected
    
    def rerank_batch(
        self,
        codes: List[str],
        candidates_dicts: List[Dict[str, List[str]]],
    ) -> List[Dict[str, str]]:
        """
        Rerank candidates for a batch of code samples.
        
        Args:
            codes: List of function codes
            candidates_dicts: List of candidate dictionaries (one per code)
            
        Returns:
            List of selected name dictionaries
        """
        results = []
        for code, cand_dict in zip(codes, candidates_dicts):
            selected = self.rerank(code, cand_dict)
            results.append(selected)
        return results


def train_reranker_contrastive(
    reranker: DualEncoderReranker,
    train_data: List[Tuple[str, str, str]],  # (context, positive_name, negative_name)
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> DualEncoderReranker:
    """
    Train reranker with contrastive InfoNCE loss.
    
    Args:
        reranker: Dual-encoder model to train
        train_data: List of (context, gold_name, wrong_name) tuples
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device to train on
        
    Returns:
        Trained reranker model
    """
    reranker = reranker.to(device)
    optimizer = torch.optim.AdamW(reranker.parameters(), lr=lr, weight_decay=0.01)
    
    # InfoNCE loss (contrastive)
    def infonce_loss(context_embeds, pos_embeds, neg_embeds, temperature):
        """
        InfoNCE contrastive loss.
        Positive pairs should have high similarity, negatives should have low similarity.
        """
        # Positive similarity
        pos_sim = (context_embeds * pos_embeds).sum(dim=-1) / temperature
        
        # Negative similarity
        neg_sim = (context_embeds * neg_embeds).sum(dim=-1) / temperature
        
        # InfoNCE: log(exp(pos) / (exp(pos) + exp(neg)))
        logits = torch.stack([pos_sim, neg_sim], dim=-1)  # (batch, 2)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss
    
    print(f"Training reranker on {device}...")
    for epoch in range(epochs):
        reranker.train()
        total_loss = 0.0
        num_batches = 0
        
        # Shuffle data
        import random
        random.shuffle(train_data)
        
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            contexts, pos_names, neg_names = zip(*batch)
            
            # Encode
            context_embeds = reranker.encode_context(list(contexts))
            pos_embeds = reranker.encode_names(list(pos_names))
            neg_embeds = reranker.encode_names(list(neg_names))
            
            # Compute loss
            loss = infonce_loss(
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
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    print("Training complete!")
    return reranker


# Example usage and utilities
def example_usage():
    """Demonstrate reranker usage."""
    
    # Initialize reranker
    print("Loading reranker...")
    reranker = DualEncoderReranker(
        model_name="microsoft/codebert-base",
        hidden_dim=768,
        dropout=0.1,
    )
    
    # Create pipeline
    pipeline = RerankerPipeline(
        reranker_model=reranker,
        context_window=100,
        collision_penalty=-5.0,
    )
    
    # Example code with placeholders
    code = """
    void processFile() {
        FILE* <ID_1> = fopen("data.txt", "r");
        int <ID_2> = 0;
        while (fgets(buffer, 256, <ID_1>)) {
            <ID_2>++;
        }
        fclose(<ID_1>);
    }
    """
    
    # Candidates from generator (top-5 per placeholder)
    candidates = {
        "<ID_1>": ["fp", "file", "handle", "fd", "stream"],
        "<ID_2>": ["count", "lines", "n", "total", "num"],
    }
    
    # Rerank and select best
    selected = pipeline.rerank(code, candidates)
    print("\nReranking results:")
    for ph, name in selected.items():
        print(f"  {ph} -> {name}")
    print(f"\nExpected: <ID_1> -> file/fp/handle, <ID_2> -> count/lines")


if __name__ == "__main__":
    example_usage()

