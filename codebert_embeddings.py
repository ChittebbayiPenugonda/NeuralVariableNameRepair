"""
CodeBERT Embedding Generator for Neural Variable Name Repair
This script converts code text into embeddings using Microsoft's CodeBERT model.

Usage Example:
    from codebert_embeddings import CodeBERTEmbedder, DataEmbeddingProcessor
    from data_pipeline import DataPipeline
    
    # Load your data
    pipeline = DataPipeline("example_output.jsonl")
    pipeline.load_data()
    input_texts, target_texts = pipeline.get_separate_arrays()
    
    # Initialize embedder and processor
    embedder = CodeBERTEmbedder()
    processor = DataEmbeddingProcessor(embedder)
    
    # Generate embeddings
    input_embeddings, target_embeddings = processor.process_from_pipeline(
        input_texts, target_texts, batch_size=8, max_length=512
    )
    
    # Save embeddings for later use
    processor.save_embeddings(input_embeddings, target_embeddings, "embeddings.npz")
    
    # Compute distances
    distances = processor.compute_distances(input_embeddings, target_embeddings)
"""

import json
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class CodeBERTEmbedder:
    """
    Generate embeddings for code using CodeBERT (microsoft/codebert-base).
    """
    
    def __init__(self, model_name: str = "microsoft/codebert-base", device: Optional[str] = None):
        """
        Initialize the CodeBERT embedder.
        
        Args:
            model_name: HuggingFace model name (default: microsoft/codebert-base)
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
        """
        print(f"Loading CodeBERT model: {model_name}")
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        print("CodeBERT model loaded successfully!")
    
    def encode_text(self, text: str, max_length: int = 512) -> np.ndarray:
        """
        Encode a single text into a CodeBERT embedding.
        
        Args:
            text: Input text/code to encode
            max_length: Maximum sequence length (default: 512)
        
        Returns:
            Numpy array of shape (hidden_size,) representing the embedding
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=max_length,
            padding='max_length',
            truncation=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding (first token) as sentence representation
            embedding = outputs.last_hidden_state[:, 0, :].squeeze()
        
        return embedding.cpu().numpy()
    
    def encode_batch(self, texts: List[str], batch_size: int = 8, max_length: int = 512) -> np.ndarray:
        """
        Encode a batch of texts into CodeBERT embeddings.
        
        Args:
            texts: List of input texts/code to encode
            batch_size: Number of texts to process at once
            max_length: Maximum sequence length (default: 512)
        
        Returns:
            Numpy array of shape (num_texts, hidden_size)
        """
        all_embeddings = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors='pt',
                max_length=max_length,
                padding='max_length',
                truncation=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embeddings
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)
    
    def compute_cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            emb1: First embedding vector
            emb2: Second embedding vector
        
        Returns:
            Cosine similarity score between -1 and 1
        """
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        return dot_product / (norm1 * norm2)
    
    def compute_euclidean_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute Euclidean distance between two embeddings.
        
        Args:
            emb1: First embedding vector
            emb2: Second embedding vector
        
        Returns:
            Euclidean distance
        """
        return np.linalg.norm(emb1 - emb2)


class DataEmbeddingProcessor:
    """
    Process data from the pipeline and generate embeddings.
    """
    
    def __init__(self, embedder: CodeBERTEmbedder):
        """
        Initialize the processor with a CodeBERT embedder.
        
        Args:
            embedder: CodeBERTEmbedder instance
        """
        self.embedder = embedder
    
    def process_from_pipeline(
        self,
        input_texts: List[str],
        target_texts: List[str],
        batch_size: int = 8,
        max_length: int = 512
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process input and target texts from the data pipeline.
        
        Args:
            input_texts: List of input code snippets with masked variables
            target_texts: List of target JSON strings (not used for embedding)
            batch_size: Batch size for processing
            max_length: Maximum sequence length
        
        Returns:
            Tuple of (input_embeddings, target_embeddings)
        """
        print(f"\nProcessing {len(input_texts)} input texts...")
        input_embeddings = self.embedder.encode_batch(
            input_texts,
            batch_size=batch_size,
            max_length=max_length
        )
        
        print(f"\nProcessing {len(target_texts)} target texts...")
        target_embeddings = self.embedder.encode_batch(
            target_texts,
            batch_size=batch_size,
            max_length=max_length
        )
        
        return input_embeddings, target_embeddings
    
    def save_embeddings(
        self,
        input_embeddings: np.ndarray,
        target_embeddings: np.ndarray,
        output_file: str
    ) -> None:
        """
        Save embeddings to a file.
        
        Args:
            input_embeddings: Input embeddings array
            target_embeddings: Target embeddings array
            output_file: Path to output file (.npz format)
        """
        output_path = Path(output_file)
        
        np.savez_compressed(
            output_path,
            input_embeddings=input_embeddings,
            target_embeddings=target_embeddings
        )
        
        print(f"\nEmbeddings saved to {output_path}")
        print(f"  Input embeddings shape: {input_embeddings.shape}")
        print(f"  Target embeddings shape: {target_embeddings.shape}")
    
    def load_embeddings(self, input_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load embeddings from a file.
        
        Args:
            input_file: Path to input .npz file
        
        Returns:
            Tuple of (input_embeddings, target_embeddings)
        """
        data = np.load(input_file)
        input_embeddings = data['input_embeddings']
        target_embeddings = data['target_embeddings']
        
        print(f"Loaded embeddings from {input_file}")
        print(f"  Input embeddings shape: {input_embeddings.shape}")
        print(f"  Target embeddings shape: {target_embeddings.shape}")
        
        return input_embeddings, target_embeddings
    
    def compute_distances(
        self,
        input_embeddings: np.ndarray,
        target_embeddings: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute pairwise distances between input and target embeddings.
        
        Args:
            input_embeddings: Input embeddings array
            target_embeddings: Target embeddings array
        
        Returns:
            Dictionary with 'cosine_similarities' and 'euclidean_distances'
        """
        assert len(input_embeddings) == len(target_embeddings), \
            "Input and target embeddings must have the same length"
        
        n_samples = len(input_embeddings)
        cosine_similarities = np.zeros(n_samples)
        euclidean_distances = np.zeros(n_samples)
        
        print("\nComputing distances...")
        for i in tqdm(range(n_samples)):
            cosine_similarities[i] = self.embedder.compute_cosine_similarity(
                input_embeddings[i],
                target_embeddings[i]
            )
            euclidean_distances[i] = self.embedder.compute_euclidean_distance(
                input_embeddings[i],
                target_embeddings[i]
            )
        
        return {
            'cosine_similarities': cosine_similarities,
            'euclidean_distances': euclidean_distances
        }
