"""
Legacy Ollama Inference Script (compatible with v0.12.x)
Runs Llama 3 8B locally on macOS.
"""

import json, argparse, subprocess
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from data_pipeline import DataPipeline
from evaluate_predictions import PredictionEvaluator


def load_prompt_template(prompt_file="prompt.txt"):
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read().strip()


def format_chat_prompt(system_prompt, user_input):
    return f"{system_prompt}\n\n{user_input}"


def query_ollama_legacy(model_name, prompt):
    """
    Works with older Ollama versions that output plain text only.
    """
    cmd = ["ollama", "run", model_name]
    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = proc.communicate(prompt.encode("utf-8"))
        if proc.returncode != 0:
            print("❌ Ollama error:", stderr.decode())
            return ""
        return stdout.decode("utf-8").strip()
    except FileNotFoundError:
        print("❌ Ollama not found or not running.")
        return ""


def run_inference(data_file, model_name="llama3:8b", prompt_file="prompt.txt",
                  output_file="llm_responses.jsonl", max_samples=None):
    print(f"\n[1/4] Loading data from {data_file}...")
    pipeline = DataPipeline(data_file)
    pipeline.load_data()
    input_texts, target_texts = pipeline.get_separate_arrays()

    if max_samples:
        input_texts, target_texts = input_texts[:max_samples], target_texts[:max_samples]
        print(f"Processing first {max_samples} samples")

    system_prompt = load_prompt_template(prompt_file)
    print(f"\n[2/4] Using Ollama model: {model_name}")

    results = []
    for i, inp in enumerate(tqdm(input_texts, desc="Generating")):
        prompt = format_chat_prompt(system_prompt, inp)
        resp = query_ollama_legacy(model_name, prompt)
        results.append({
            "index": i,
            "input_text": inp,
            "target_text": target_texts[i],
            "llm_response": resp,
        })

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n✅ Saved {len(results)} responses to {output_file}")
    return results


def evaluate_results(results, output_eval_file="evaluation_results.json"):
    print("\n" + "=" * 80)
    print(" EVALUATING PREDICTIONS")
    print("=" * 80)
    llm_responses = [r["llm_response"] for r in results]
    target_texts = [r["target_text"] for r in results]
    evaluator = PredictionEvaluator()
    cleaned = evaluator.prepare_llm_responses(llm_responses)
    metrics = evaluator.evaluate(cleaned, target_texts)
    evaluator.print_report(metrics, show_details=False)
    evaluator.save_results(metrics, output_eval_file)
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="example_output.jsonl")
    p.add_argument("--model", default="llama3:8b")
    p.add_argument("--prompt", default="prompt.txt")
    p.add_argument("--output", default="llm_responses.jsonl")
    p.add_argument("--eval-output", default="evaluation_results.json")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--skip-eval", action="store_true")
    args = p.parse_args()

    results = run_inference(args.data, args.model, args.prompt,
                            args.output, args.max_samples)
    if not args.skip_eval:
        evaluate_results(results, args.eval_output)
    print("\n" + "=" * 80)
    print(" COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
