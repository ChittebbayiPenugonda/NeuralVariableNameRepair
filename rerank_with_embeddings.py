"""
Rerank LLM variable-name candidates using the dual-encoder CodeBERT model (H3).

Expected input JSONL (one record per example):
{
  "input_text": "... masked C++ code ...",
  "target_text": "{...}",                     # ground truth mapping (string)
  "llm_response": "{...}",                    # (optional) single best mapping from generator
  "candidate_responses": ["{...}", "{...}"]   # list of JSON candidate mappings (top-k)
}

You can produce `candidate_responses` by modifying run_inference.py (vLLM) to
generate n>1 samples per input. This script then chooses, for each <ID_i>,
the candidate name whose embedding is closest to the code embedding.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from finetune_codebert_name_embedding import CodeToNameEmbeddingModel
from evaluate_predictions import PredictionEvaluator


def load_results_from_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    print(f"Loaded {len(records)} records from {path}")
    return records


def parse_candidates(record: Dict[str, Any]) -> List[str]:
    """
    Return list of candidate JSON strings for this record.
    If 'candidate_responses' is missing, fall back to ['llm_response'].
    """
    if "candidate_responses" in record and record["candidate_responses"]:
        return record["candidate_responses"]
    elif "llm_response" in record and record["llm_response"]:
        return [record["llm_response"]]
    else:
        return []


def build_placeholder_candidates(candidates: List[str]) -> Dict[str, List[str]]:
    """
    From a list of JSON mapping strings, build:
      placeholder -> list of candidate variable names (deduplicated, in order).
    """
    placeholder_to_names: Dict[str, List[str]] = {}

    for cand_json in candidates:
        try:
            mapping = json.loads(cand_json)
            if not isinstance(mapping, dict):
                continue
        except Exception:
            continue

        for placeholder, name in mapping.items():
            if not isinstance(name, str):
                continue
            name = name.strip()
            if not name:
                continue
            lst = placeholder_to_names.setdefault(placeholder, [])
            if name not in lst:
                lst.append(name)

    return placeholder_to_names


def load_dual_encoder(checkpoint_path: str):
    """
    Load the dual-encoder checkpoint produced by finetune_codebert_name_embedding.py.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model_name = ckpt["model_name"]
    state_dict = ckpt["model_state_dict"]

    model = CodeToNameEmbeddingModel(model_name=model_name)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer, device


def embed_code(model, tokenizer, device, code: str, max_length: int = 256) -> torch.Tensor:
    enc = tokenizer(
        code,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        cls = model.encode(input_ids=input_ids, attention_mask=attention_mask)
        code_emb = F.normalize(model.code_proj(cls), p=2, dim=-1)

    # return (hidden,) vector
    return code_emb[0]


def embed_names(
    model,
    tokenizer,
    device,
    names: List[str],
    max_length: int = 64,
) -> torch.Tensor:
    enc = tokenizer(
        names,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        cls = model.encode(input_ids=input_ids, attention_mask=attention_mask)
        name_emb = F.normalize(model.name_proj(cls), p=2, dim=-1)

    # shape: (num_names, hidden)
    return name_emb


def rerank_record(
    model,
    tokenizer,
    device,
    record: Dict[str, Any],
    max_code_len: int = 256,
) -> Tuple[str, Dict[str, str]]:
    """
    Rerank candidate variable names for a single record.

    Returns:
      - reranked_json_str: JSON string of best names per placeholder.
      - chosen_mapping: dict mapping placeholder -> best name.
    """
    code = record.get("input_text", "")
    candidates = parse_candidates(record)
    if not code or not candidates:
        # Nothing to do; return first candidate or empty
        if candidates:
            return candidates[0], {}
        else:
            return "{}", {}

    placeholder_to_names = build_placeholder_candidates(candidates)
    if not placeholder_to_names:
        return candidates[0], {}

    code_emb = embed_code(model, tokenizer, device, code, max_length=max_code_len)

    chosen_mapping: Dict[str, str] = {}

    for placeholder, name_list in placeholder_to_names.items():
        if not name_list:
            continue

        name_embs = embed_names(model, tokenizer, device, name_list)
        # (hidden,) · (num_names, hidden)^T -> (num_names,)
        scores = torch.matmul(name_embs, code_emb.unsqueeze(-1)).squeeze(-1)
        best_idx = int(torch.argmax(scores).item())
        chosen_mapping[placeholder] = name_list[best_idx]

    reranked_json_str = json.dumps(chosen_mapping, ensure_ascii=False)
    return reranked_json_str, chosen_mapping


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rerank LLM variable-name candidates with dual-encoder embeddings"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="JSONL file with LLM outputs (must contain candidate_responses or llm_response)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to dual_encoder_codebert.pt checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="llm_responses_reranked.jsonl",
        help="Where to write reranked JSONL records",
    )
    parser.add_argument(
        "--eval_output",
        type=str,
        default="evaluation_results_reranked.json",
        help="Where to write evaluation metrics JSON",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    records = load_results_from_jsonl(args.input)
    model, tokenizer, device = load_dual_encoder(args.checkpoint)

    reranked_records: List[Dict[str, Any]] = []
    baseline_predictions: List[str] = []
    reranked_predictions: List[str] = []
    target_texts: List[str] = []

    for rec in records:
        reranked_json, _ = rerank_record(model, tokenizer, device, rec)
        new_rec = dict(rec)
        new_rec["reranked_response"] = reranked_json
        reranked_records.append(new_rec)

        # For evaluation
        baseline = rec.get("llm_response")
        if baseline is None:
            # fall back to first candidate if any
            cands = parse_candidates(rec)
            baseline = cands[0] if cands else "{}"

        baseline_predictions.append(baseline)
        reranked_predictions.append(reranked_json)
        target_texts.append(rec.get("target_text", "{}"))

    # Save reranked JSONL
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in reranked_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Saved reranked records to {args.output}")

    # Evaluate baseline vs reranked
    evaluator = PredictionEvaluator()

    base_clean = evaluator.prepare_llm_responses(baseline_predictions)
    rerank_clean = evaluator.prepare_llm_responses(reranked_predictions)

    base_metrics = evaluator.evaluate(base_clean, target_texts)
    rerank_metrics = evaluator.evaluate(rerank_clean, target_texts)

    print("\n=== Baseline (original LLM response) ===")
    evaluator.print_report(base_metrics, show_details=False)

    print("\n=== After reranking with dual encoder ===")
    evaluator.print_report(rerank_metrics, show_details=False)

    # Save rerank metrics
    Path(args.eval_output).parent.mkdir(parents=True, exist_ok=True)
    evaluator.save_results(
        {
            "baseline": base_metrics,
            "reranked": rerank_metrics,
        },
        args.eval_output,
    )
    print(f"Saved evaluation metrics to {args.eval_output}")


if __name__ == "__main__":
    main()
