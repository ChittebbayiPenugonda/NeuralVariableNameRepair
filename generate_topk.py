"""
Generate Top-K Candidates from Fine-tuned Generator

This script generates multiple candidate names per placeholder using the fine-tuned
LLM generator. These candidates are then used for reranking.

Usage:
    python generate_topk.py \
        --data_path ./test_data.jsonl \
        --base_model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --adapter_path ./lora-llama3-mapping \
        --output_file ./generator_top_k.jsonl \
        --k 10 --num_samples 20
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from data_pipeline import DataPipeline


SYSTEM_PROMPT = (
    "You are a coder that maps placeholder identifiers in C/C++ code to clear variable names.\n"
    "Only output a valid JSON object mapping placeholders to names, e.g.:\n"
    "{\"<ID_1>\": \"x\", \"<ID_2>\": \"count\"}\n"
    "Do not add any extra text."
)


def build_chat_prompt(user_input: str, tokenizer) -> str:
    """Build chat prompt using tokenizer's chat template."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


def extract_json_from_text(text: str) -> Dict[str, str]:
    """
    Extract JSON object from LLM response.
    Handles code fences and extraneous text.
    """
    text = text.strip()
    
    # Remove code fences
    if '```json' in text:
        start = text.find('```json') + 7
        end = text.find('```', start)
        if end != -1:
            text = text[start:end].strip()
    elif '```' in text:
        start = text.find('```') + 3
        end = text.find('```', start)
        if end != -1:
            text = text[start:end].strip()
    
    # Extract first balanced JSON object
    first_brace = text.find('{')
    if first_brace == -1:
        return {}
    
    # Find matching closing brace
    stack = []
    for i in range(first_brace, len(text)):
        if text[i] == '{':
            stack.append(i)
        elif text[i] == '}':
            stack.pop()
            if not stack:
                json_text = text[first_brace:i+1]
                try:
                    return json.loads(json_text)
                except json.JSONDecodeError:
                    return {}
    
    return {}


def extract_placeholders(code: str) -> List[str]:
    """Extract all placeholders from code (e.g., <ID_1>, <ID_2>)."""
    pattern = r'<ID_\d+>'
    placeholders = re.findall(pattern, code)
    return sorted(list(set(placeholders)))


def generate_topk_for_sample(
    code: str,
    model,
    tokenizer,
    k: int = 10,
    num_samples: int = 20,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_new_tokens: int = 256,
) -> Dict[str, List[str]]:
    """
    Generate top-k candidate names for each placeholder in code.
    
    Strategy:
    - Generate num_samples diverse responses using sampling
    - Extract placeholder->name mappings from each response
    - For each placeholder, collect all suggested names and keep top-k by frequency
    
    Args:
        code: Code snippet with placeholders
        model: Fine-tuned generator model
        tokenizer: Tokenizer
        k: Number of candidates to keep per placeholder
        num_samples: Number of samples to generate (more samples = more diversity)
        temperature: Sampling temperature (higher = more diverse)
        top_p: Nucleus sampling threshold
        max_new_tokens: Max tokens to generate
        
    Returns:
        Dict mapping placeholder -> list of top-k candidate names
    """
    # Build prompt
    prompt = build_chat_prompt(code, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Extract placeholders
    placeholders = extract_placeholders(code)
    
    # Collect candidates from multiple samples
    candidate_counts = {ph: {} for ph in placeholders}
    
    # Generate multiple samples
    for _ in range(num_samples):
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        # Decode
        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
        )
        
        # Extract JSON mapping
        mapping = extract_json_from_text(response)
        
        # Count candidates for each placeholder
        for ph, name in mapping.items():
            if ph in candidate_counts:
                name = name.strip()
                if name:  # Only count non-empty names
                    candidate_counts[ph][name] = candidate_counts[ph].get(name, 0) + 1
    
    # Select top-k by frequency for each placeholder
    topk_candidates = {}
    for ph in placeholders:
        counts = candidate_counts[ph]
        if counts:
            # Sort by frequency (descending)
            sorted_candidates = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            topk = [name for name, _ in sorted_candidates[:k]]
        else:
            # Fallback: generic name if no valid candidates found
            topk = [f"var{ph.replace('<ID_', '').replace('>', '')}"]
        
        topk_candidates[ph] = topk
    
    return topk_candidates


def generate_topk_batch(
    data_file: str,
    base_model: str,
    adapter_path: str,
    output_file: str,
    k: int = 10,
    num_samples: int = 20,
    temperature: float = 0.8,
    max_samples: int = None,
) -> None:
    """
    Generate top-k candidates for all samples in dataset.
    
    Args:
        data_file: Input JSONL file with masked code
        base_model: Base model name
        adapter_path: Path to LoRA adapter checkpoint
        output_file: Output JSONL file for top-k candidates
        k: Number of candidates per placeholder
        num_samples: Number of samples to generate for diversity
        temperature: Sampling temperature
        max_samples: Max number of samples to process (for testing)
    """
    # Load data
    print(f"\n[1/4] Loading data from {data_file}...")
    pipeline = DataPipeline(data_file)
    pipeline.load_data()
    data = pipeline.get_full_format()
    
    if max_samples:
        data = data[:max_samples]
        print(f"Processing first {max_samples} samples")
    
    # Load model
    print(f"\n[2/4] Loading model...")
    print(f"  Base model: {base_model}")
    print(f"  Adapter: {adapter_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # Load adapter
    if os.path.exists(adapter_path):
        print(f"  Loading adapter from {adapter_path}")
        model = PeftModel.from_pretrained(base, adapter_path)
    else:
        print(f"  Warning: Adapter not found at {adapter_path}")
        print(f"  Using base model only (not recommended)")
        model = base
    
    model.eval()
    
    # Generate top-k for each sample
    print(f"\n[3/4] Generating top-{k} candidates...")
    print(f"  Samples per input: {num_samples}")
    print(f"  Temperature: {temperature}")
    
    results = []
    
    for idx, sample in enumerate(tqdm(data, desc="Generating")):
        code = sample['input_text']
        target = sample['target_text']
        
        # Generate top-k candidates
        topk = generate_topk_for_sample(
            code,
            model,
            tokenizer,
            k=k,
            num_samples=num_samples,
            temperature=temperature,
        )
        
        # Save result
        result = {
            'index': idx,
            'input_text': code,
            'target_text': target,
            'top_k_predictions': topk,
            'k': k,
            'num_samples': num_samples,
        }
        results.append(result)
    
    # Save results
    print(f"\n[4/4] Saving results to {output_file}...")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"\n✓ Saved {len(results)} results with top-{k} candidates")
    
    # Print example
    if results:
        print("\nExample result:")
        ex = results[0]
        print(f"  Sample 0:")
        for ph, candidates in ex['top_k_predictions'].items():
            print(f"    {ph}: {candidates[:3]}... (showing first 3)")


def main():
    parser = argparse.ArgumentParser(description='Generate top-k candidates from fine-tuned model')
    
    parser.add_argument('--data_path', type=str, required=True,
                        help='Input JSONL file with masked code')
    parser.add_argument('--base_model', type=str, 
                        default='meta-llama/Meta-Llama-3.1-8B-Instruct',
                        help='Base model name')
    parser.add_argument('--adapter_path', type=str, required=True,
                        help='Path to LoRA adapter checkpoint')
    parser.add_argument('--output_file', type=str, default='generator_top_k.jsonl',
                        help='Output JSONL file for top-k candidates')
    
    parser.add_argument('--k', type=int, default=10,
                        help='Number of candidates to keep per placeholder')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of samples to generate (more = more diversity)')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature (higher = more diverse)')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Nucleus sampling threshold')
    parser.add_argument('--max_new_tokens', type=int, default=256,
                        help='Max tokens to generate')
    
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max number of samples to process (for testing)')
    
    args = parser.parse_args()
    
    generate_topk_batch(
        data_file=args.data_path,
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        output_file=args.output_file,
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

