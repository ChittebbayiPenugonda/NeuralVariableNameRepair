"""
Evaluation Pipeline for LLM Predictions
Scores LLM outputs against ground truth targets.

This version supports PARTIAL CREDIT for JSON mappings:
- If both prediction and target parse as JSON objects, we compare values for each key
  present in the TARGET (e.g., "<ID_1>", "<ID_2>", ...).
- Score for a sample = (# keys correct) / (# target keys)
- We also report micro-averaged accuracy across all samples:
    total_correct_keys / total_target_keys
If JSON parsing fails, we fall back to exact string match (0/1).
"""

import json
from typing import List, Dict, Tuple, Optional


class PredictionEvaluator:
    """
    Evaluator for comparing LLM predictions against ground truth.
    Supports exact match and partial-credit JSON mapping scoring.
    """

    def __init__(self):
        self.results = []

    # -------------------------- Parsing Helpers --------------------------

    def parse_llm_response(self, response: str) -> str:
        """
        Extract a JSON substring if present (handles ```json blocks, etc.)
        """
        response = response.strip()

        # Try to extract JSON inside code fences
        if '```json' in response:
            start = response.find('```json') + 7
            end = response.find('```', start)
            if end != -1:
                response = response[start:end].strip()
        elif '```' in response:
            start = response.find('```') + 3
            end = response.find('```', start)
            if end != -1:
                response = response[start:end].strip()

        # Trim to first {...last}
        first_brace = response.find('{')
        last_brace = response.rfind('}')
        if first_brace != -1 and last_brace != -1:
            response = response[first_brace:last_brace + 1]

        return response

    def prepare_llm_responses(self, raw_responses: List[str]) -> List[str]:
        return [self.parse_llm_response(resp) for resp in raw_responses]

    def _try_load_json(self, s: str) -> Optional[Dict]:
        """
        Try to parse JSON object from string. Return dict or None.
        """
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        return None

    # -------------------------- Scoring Methods --------------------------

    def exact_match_score(self, prediction: str, target: str) -> int:
        """
        Legacy exact string equality (0/1).
        """
        return 1 if prediction.strip() == target.strip() else 0

    def mapping_partial_score(self, pred_str: str, tgt_str: str) -> Optional[Dict]:
        """
        If both strings parse as JSON dicts, compute partial-credit mapping score.
        We compare ONLY keys that exist in the TARGET mapping.
        Returns a dict with detailed counts, or None if parsing failed.
        """
        pred_obj = self._try_load_json(pred_str)
        tgt_obj = self._try_load_json(tgt_str)
        if pred_obj is None or tgt_obj is None:
            return None

        # Count over target keys
        total = len(tgt_obj)
        correct = 0
        per_key = []

        for k, tgt_val in tgt_obj.items():
            pred_has = (k in pred_obj)
            pred_val = pred_obj.get(k, None)
            is_correct = pred_has and (str(pred_val).strip() == str(tgt_val).strip())
            if is_correct:
                correct += 1
            per_key.append({
                "key": k,
                "target": tgt_val,
                "pred": pred_val,
                "match": is_correct
            })

        # Track "extra" keys present only in prediction (not counted against score, but reported)
        extra_keys = [k for k in pred_obj.keys() if k not in tgt_obj]

        return {
            "correct_keys": correct,
            "total_target_keys": total,
            "per_key": per_key,
            "extra_pred_keys": extra_keys,
        }

    def evaluate(self, predictions: List[str], targets: List[str]) -> Dict:
        """
        Evaluate predictions vs targets.
        - If both parse as JSON dicts -> partial credit per sample.
        - Else -> fallback to exact match (0/1) for that sample.
        Returns summary with micro-averaged accuracy across keys.
        """
        if len(predictions) != len(targets):
            raise ValueError(f"Mismatch: {len(predictions)} predictions vs {len(targets)} targets")

        self.results = []
        total_trials = len(targets)

        # Micro-averaging over all keys across all samples
        global_correct_keys = 0
        global_total_keys = 0

        exact_match_correct = 0  # count of exact matches used for non-JSON cases
        num_partial = 0
        num_exact_fallback = 0

        for i, (pred, tgt) in enumerate(zip(predictions, targets)):
            partial = self.mapping_partial_score(pred, tgt)
            if partial is not None:
                # Partial-credit JSON mapping
                num_partial += 1
                c = partial["correct_keys"]
                t = partial["total_target_keys"]
                global_correct_keys += c
                global_total_keys += t
                sample_acc = (c / t) if t > 0 else 0.0
                self.results.append({
                    "trial_id": i,
                    "mode": "partial_json",
                    "prediction": pred,
                    "target": tgt,
                    "correct_keys": c,
                    "total_target_keys": t,
                    "sample_accuracy": sample_acc,
                    "per_key": partial["per_key"],
                    "extra_pred_keys": partial["extra_pred_keys"],
                })
            else:
                # Fallback: exact match (counts as 0/1 over ONE "virtual key")
                num_exact_fallback += 1
                s = self.exact_match_score(pred, tgt)
                exact_match_correct += s
                global_correct_keys += s
                global_total_keys += 1  # treat as one virtual key
                self.results.append({
                    "trial_id": i,
                    "mode": "exact_fallback",
                    "prediction": pred,
                    "target": tgt,
                    "score": s,
                    "sample_accuracy": float(s),  # 1.0 or 0.0
                })

        micro_accuracy = (global_correct_keys / global_total_keys) if global_total_keys > 0 else 0.0

        return {
            "total_trials": total_trials,
            "num_partial_json": num_partial,
            "num_exact_fallback": num_exact_fallback,
            "global_correct_keys": global_correct_keys,
            "global_total_keys": global_total_keys,
            "micro_accuracy": micro_accuracy,
            "micro_accuracy_percentage": 100.0 * micro_accuracy,
            "exact_fallback_correct": exact_match_correct,
        }

    def get_detailed_results(self) -> List[Dict]:
        return self.results

    # -------------------------- Reporting & Saving --------------------------

    def print_report(self, metrics: Dict, show_details: bool = True):
        print("\n" + "=" * 80)
        print(" EVALUATION REPORT (Partial Credit Enabled)")
        print("=" * 80)

        print(f"\nTotal Samples:           {metrics['total_trials']}")
        print(f"Samples (partial JSON):  {metrics['num_partial_json']}")
        print(f"Samples (exact fallback):{metrics['num_exact_fallback']}")
        print(f"\nTotal Correct Keys:      {metrics['global_correct_keys']}")
        print(f"Total Target Keys:       {metrics['global_total_keys']}")
        print(f"Micro Accuracy:          {metrics['micro_accuracy']:.4f} ({metrics['micro_accuracy_percentage']:.2f}%)")

        if show_details and self.results:
            print("\n" + "-" * 80)
            print(" TRIAL-BY-TRIAL RESULTS")
            print("-" * 80)
            for r in self.results:
                if r["mode"] == "partial_json":
                    print(f"\nTrial {r['trial_id']}: PARTIAL ({r['correct_keys']}/{r['total_target_keys']}) "
                          f"= {r['sample_accuracy']:.3f}")
                    # Optional concise per-key line:
                    misses = [pk for pk in r["per_key"] if not pk["match"]]
                    if misses:
                        miss_str = ", ".join([f"{m['key']}→{m['pred']} (target {m['target']})" for m in misses])
                        print(f"  Misses: {miss_str}")
                    if r["extra_pred_keys"]:
                        print(f"  Extra prediction-only keys: {r['extra_pred_keys']}")
                else:
                    print(f"\nTrial {r['trial_id']}: EXACT-FALLBACK "
                          f"= {r['sample_accuracy']:.3f} (Prediction == Target)")

        print("\n" + "=" * 80)

    def save_results(self, metrics: Dict, output_file: str):
        output = {
            'summary': metrics,
            'detailed_results': self.results
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {output_file}")
