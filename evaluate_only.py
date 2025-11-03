"""
Evaluate existing LLM responses using PredictionEvaluator.
Usage:
    python evaluate_only.py --input llm_responses.jsonl --output evaluation_results.json
"""

import json, argparse
from evaluate_predictions import PredictionEvaluator


def load_results_from_jsonl(path):
    """Load response records from an existing JSONL file."""
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="llm_responses.jsonl",
                        help="Existing responses JSONL file")
    parser.add_argument("--output", default="evaluation_results.json",
                        help="Where to save evaluation results")
    args = parser.parse_args()

    # Load saved predictions
    results = load_results_from_jsonl(args.input)

    # Extract relevant fields
    llm_responses = [r["llm_response"] for r in results]
    target_texts = [r["target_text"] for r in results]

    # Run evaluation
    evaluator = PredictionEvaluator()
    cleaned = evaluator.prepare_llm_responses(llm_responses)
    metrics = evaluator.evaluate(cleaned, target_texts)
    evaluator.print_report(metrics, show_details=True)
    evaluator.save_results(metrics, args.output)


if __name__ == "__main__":
    main()
