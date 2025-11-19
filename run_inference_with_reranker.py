"""
End-to-End Inference with Generator + Reranker

This script runs the full two-stage pipeline:
1. Generator: Fine-tuned LLM produces top-k candidates per placeholder
2. Reranker: Dual-encoder scores and selects best candidate

Usage:
    python run_inference_with_reranker.py \
        --data_path ./test_data.jsonl \
        --base_model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --adapter_path ./lora-llama3-mapping \
        --reranker_checkpoint ./reranker_checkpoint/best_model.pt \
        --output_file ./predictions_reranked.jsonl \
        --k 10
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from data_pipeline import DataPipeline
from evaluate_predictions import PredictionEvaluator
from reranker import DualEncoderReranker, RerankerPipeline
from generate_topk import (
    generate_topk_for_sample,
    extract_json_from_text,
    SYSTEM_PROMPT,
)


def load_reranker(
    checkpoint_path: str,
    base_model: str = "microsoft/codebert-base",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> DualEncoderReranker:
    """
    Load trained reranker from checkpoint.
    
    Args:
        checkpoint_path: Path to .pt checkpoint file
        base_model: Base encoder model name
        device: Device to load on
        
    Returns:
        Loaded reranker model
    """
    print(f"Loading reranker from {checkpoint_path}...")
    
    # Initialize model
    reranker = DualEncoderReranker(
        model_name=base_model,
        hidden_dim=768,
        dropout=0.1,
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    reranker.load_state_dict(checkpoint['model_state_dict'])
    reranker = reranker.to(device)
    reranker.eval()
    
    print(f"  Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"  Val loss: {checkpoint.get('val_loss', 'N/A')}")
    
    return reranker


def run_generator_and_reranker(
    data_file: str,
    base_model: str,
    adapter_path: str,
    reranker_checkpoint: str,
    output_file: str,
    eval_output_file: str,
    k: int = 10,
    num_samples: int = 20,
    temperature: float = 0.8,
    max_samples: int = None,
) -> None:
    """
    Run full pipeline: generator top-k + reranker selection.
    
    Args:
        data_file: Input JSONL with masked code
        base_model: Base generator model
        adapter_path: LoRA adapter path
        reranker_checkpoint: Reranker checkpoint path
        output_file: Output file for predictions
        eval_output_file: Output file for evaluation results
        k: Top-k candidates from generator
        num_samples: Samples per input for diversity
        temperature: Sampling temperature
        max_samples: Max samples to process (for testing)
    """
    
    # === STAGE 1: Load Data ===
    print("\n" + "="*80)
    print("STAGE 1: LOADING DATA")
    print("="*80)
    
    pipeline = DataPipeline(data_file)
    pipeline.load_data()
    data = pipeline.get_full_format()
    
    if max_samples:
        data = data[:max_samples]
        print(f"Processing first {max_samples} samples")
    
    # === STAGE 2: Load Generator ===
    print("\n" + "="*80)
    print("STAGE 2: LOADING GENERATOR")
    print("="*80)
    print(f"  Base model: {base_model}")
    print(f"  Adapter: {adapter_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    if os.path.exists(adapter_path):
        generator = PeftModel.from_pretrained(base, adapter_path)
    else:
        print(f"  Warning: Adapter not found, using base model only")
        generator = base
    
    generator.eval()
    
    # === STAGE 3: Load Reranker ===
    print("\n" + "="*80)
    print("STAGE 3: LOADING RERANKER")
    print("="*80)
    
    reranker_model = load_reranker(
        reranker_checkpoint,
        base_model="microsoft/codebert-base",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    reranker_pipeline = RerankerPipeline(
        reranker_model=reranker_model,
        context_window=100,
        collision_penalty=-5.0,
    )
    
    # === STAGE 4: Generate + Rerank ===
    print("\n" + "="*80)
    print("STAGE 4: GENERATING & RERANKING")
    print("="*80)
    print(f"  Top-k: {k}")
    print(f"  Samples per input: {num_samples}")
    print(f"  Temperature: {temperature}\n")
    
    results = []
    
    for idx, sample in enumerate(tqdm(data, desc="Processing")):
        code = sample['input_text']
        target = sample['target_text']
        
        # Step 4a: Generate top-k candidates
        topk_candidates = generate_topk_for_sample(
            code,
            generator,
            tokenizer,
            k=k,
            num_samples=num_samples,
            temperature=temperature,
        )
        
        # Step 4b: Rerank and select best
        reranked_mapping = reranker_pipeline.rerank(code, topk_candidates)
        
        # Convert to JSON string for evaluation
        prediction_json = json.dumps(reranked_mapping, sort_keys=True)
        
        # Save result
        result = {
            'index': idx,
            'input_text': code,
            'target_text': target,
            'llm_response': prediction_json,  # For evaluator compatibility
            'top_k_candidates': topk_candidates,
            'reranked_selection': reranked_mapping,
        }
        results.append(result)
    
    # === STAGE 5: Save Predictions ===
    print("\n" + "="*80)
    print("STAGE 5: SAVING PREDICTIONS")
    print("="*80)
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"✓ Saved {len(results)} predictions to {output_file}")
    
    # === STAGE 6: Evaluate ===
    print("\n" + "="*80)
    print("STAGE 6: EVALUATING PREDICTIONS")
    print("="*80)
    
    evaluator = PredictionEvaluator()
    
    # Extract responses and targets
    predictions = [r['llm_response'] for r in results]
    targets = [r['target_text'] for r in results]
    
    # Evaluate
    cleaned_predictions = evaluator.prepare_llm_responses(predictions)
    metrics = evaluator.evaluate(cleaned_predictions, targets)
    
    # Print report
    evaluator.print_report(metrics, show_details=False)
    
    # Save evaluation
    evaluator.save_results(metrics, eval_output_file)
    
    # === Summary ===
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Predictions: {output_file}")
    print(f"Evaluation: {eval_output_file}")
    print(f"\nExact Match: {metrics['micro_accuracy_percentage']:.2f}%")
    print(f"Correct Keys: {metrics['global_correct_keys']}/{metrics['global_total_keys']}")
    
    # Print example
    if results:
        print("\n" + "-"*80)
        print("EXAMPLE PREDICTION")
        print("-"*80)
        ex = results[0]
        print(f"Input placeholders: {list(ex['top_k_candidates'].keys())}")
        print(f"\nTop-3 candidates per placeholder:")
        for ph, cands in ex['top_k_candidates'].items():
            print(f"  {ph}: {cands[:3]}")
        print(f"\nReranked selection:")
        for ph, name in ex['reranked_selection'].items():
            print(f"  {ph} → {name}")
        print(f"\nGold mapping: {ex['target_text']}")


def main():
    parser = argparse.ArgumentParser(
        description='Run inference with generator + reranker pipeline'
    )
    
    # Data
    parser.add_argument('--data_path', type=str, required=True,
                        help='Input JSONL file with masked code')
    parser.add_argument('--output_file', type=str, default='predictions_reranked.jsonl',
                        help='Output file for predictions')
    parser.add_argument('--eval_output', type=str, default='evaluation_reranked.json',
                        help='Output file for evaluation results')
    
    # Generator
    parser.add_argument('--base_model', type=str,
                        default='meta-llama/Meta-Llama-3.1-8B-Instruct',
                        help='Base generator model')
    parser.add_argument('--adapter_path', type=str, required=True,
                        help='Path to LoRA adapter checkpoint')
    
    # Reranker
    parser.add_argument('--reranker_checkpoint', type=str, required=True,
                        help='Path to reranker checkpoint (.pt file)')
    parser.add_argument('--reranker_base', type=str, default='microsoft/codebert-base',
                        help='Base model for reranker encoders')
    
    # Generation config
    parser.add_argument('--k', type=int, default=10,
                        help='Number of candidates per placeholder')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of samples for diversity')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature')
    
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max samples to process (for testing)')
    
    args = parser.parse_args()
    
    run_generator_and_reranker(
        data_file=args.data_path,
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        reranker_checkpoint=args.reranker_checkpoint,
        output_file=args.output_file,
        eval_output_file=args.eval_output,
        k=args.k,
        num_samples=args.num_samples,
        temperature=args.temperature,
        max_samples=args.max_samples,
    )
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

