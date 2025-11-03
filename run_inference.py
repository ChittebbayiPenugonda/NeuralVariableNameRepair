"""
LLM Inference Script using vLLM and Llama 3.1 8B Instruct
Generates variable name predictions for masked C++ code.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from vllm import LLM, SamplingParams
from data_pipeline import DataPipeline
from evaluate_predictions import PredictionEvaluator


def load_prompt_template(prompt_file: str = "prompt.txt") -> str:
    """Load the prompt template from file."""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read().strip()


def format_chat_prompt(system_prompt: str, user_input: str) -> str:
    """
    Format prompt for Llama 3.1 Instruct using chat template format.
    
    Args:
        system_prompt: System instruction for the task
        user_input: The masked code snippet
        
    Returns:
        Formatted prompt string
    """
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def run_inference(
    data_file: str,
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    prompt_file: str = "prompt.txt",
    output_file: str = "llm_responses.jsonl",
    max_samples: int = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    tensor_parallel_size: int = 1,
) -> List[Dict]:
    """
    Run LLM inference on the dataset using vLLM.
    
    Args:
        data_file: Path to input JSONL file
        model_name: HuggingFace model identifier
        prompt_file: Path to prompt template file
        output_file: Path to save responses
        max_samples: Maximum number of samples to process (None for all)
        temperature: Sampling temperature (0.0 for greedy)
        max_tokens: Maximum tokens to generate
        tensor_parallel_size: Number of GPUs for tensor parallelism
        
    Returns:
        List of response dictionaries
    """
    
    # Load data
    print(f"\n[1/4] Loading data from {data_file}...")
    pipeline = DataPipeline(data_file)
    pipeline.load_data()
    input_texts, target_texts = pipeline.get_separate_arrays()
    
    if max_samples:
        input_texts = input_texts[:max_samples]
        target_texts = target_texts[:max_samples]
        print(f"Processing first {max_samples} samples")
    
    # Load prompt template
    print(f"\n[2/4] Loading prompt template from {prompt_file}...")
    system_prompt = load_prompt_template(prompt_file)
    
    # Initialize vLLM
    print(f"\n[3/4] Initializing vLLM with {model_name}...")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        dtype="auto",
        max_model_len=4096,
    )
    
    # Prepare sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.95 if temperature > 0 else 1.0,
        max_tokens=max_tokens,
        stop=["<|eot_id|>", "<|end_of_text|>"],
    )
    
    # Format prompts
    print("\n[4/4] Generating predictions...")
    prompts = [format_chat_prompt(system_prompt, input_text) for input_text in input_texts]
    
    # Run inference
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    
    # Collect responses
    results = []
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text.strip()
        results.append({
            'index': i,
            'input_text': input_texts[i],
            'target_text': target_texts[i],
            'llm_response': generated_text,
        })
    
    # Save responses
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"\n Saved {len(results)} responses to {output_file}")
    
    return results


def evaluate_results(results: List[Dict], output_eval_file: str = "evaluation_results.json"):
    """
    Evaluate LLM responses against ground truth.
    
    Args:
        results: List of result dictionaries with llm_response and target_text
        output_eval_file: Path to save evaluation results
    """
    print("\n" + "="*80)
    print(" EVALUATING PREDICTIONS")
    print("="*80)
    
    # Extract responses and targets
    llm_responses = [r['llm_response'] for r in results]
    target_texts = [r['target_text'] for r in results]
    
    # Evaluate
    evaluator = PredictionEvaluator()
    cleaned_responses = evaluator.prepare_llm_responses(llm_responses)
    metrics = evaluator.evaluate(cleaned_responses, target_texts)
    
    # Print report
    evaluator.print_report(metrics, show_details=False)
    
    # Save results
    evaluator.save_results(metrics, output_eval_file)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Run LLM inference for variable name prediction')
    parser.add_argument('--data', type=str, default='example_output.jsonl',
                        help='Input JSONL file with masked code')
    parser.add_argument('--model', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct',
                        help='HuggingFace model name')
    parser.add_argument('--prompt', type=str, default='prompt.txt',
                        help='Path to prompt template file')
    parser.add_argument('--output', type=str, default='llm_responses.jsonl',
                        help='Output file for LLM responses')
    parser.add_argument('--eval-output', type=str, default='evaluation_results.json',
                        help='Output file for evaluation results')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to process')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Sampling temperature (0.0 for greedy)')
    parser.add_argument('--max-tokens', type=int, default=512,
                        help='Maximum tokens to generate')
    parser.add_argument('--tensor-parallel-size', type=int, default=1,
                        help='Number of GPUs for tensor parallelism')
    parser.add_argument('--skip-eval', action='store_true',
                        help='Skip evaluation step')
    
    args = parser.parse_args()
    
    # Run inference
    results = run_inference(
        data_file=args.data,
        model_name=args.model,
        prompt_file=args.prompt,
        output_file=args.output,
        max_samples=args.max_samples,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    
    # Evaluate
    if not args.skip_eval:
        evaluate_results(results, args.eval_output)
    
    print("\n" + "="*80)
    print(" COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()

