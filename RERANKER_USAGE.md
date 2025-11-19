# Reranker Usage Guide

## Quick Summary

**What is it?** A two-stage system that improves variable name predictions:
1. Generator makes top-10 candidates
2. Reranker picks the best one

**Why?** Improves accuracy by +5-10% (Exact Match)

---

## 3-Step Workflow

### Step 1: Fine-tune Generator (if not done)
```bash
python finetune_lora_llama.py \
    --data_path ./train_data.jsonl \
    --base_model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --out_dir ./lora-llama3-mapping \
    --epochs 3 --lr 2e-4
```

### Step 2: Train Reranker

**2a. Generate top-k candidates:**
```bash
python generate_topk.py \
    --data_path ./train_data.jsonl \
    --adapter_path ./lora-llama3-mapping \
    --output_file ./generator_top_k_train.jsonl \
    --k 10 --num_samples 20
```

**2b. Train reranker:**
```bash
python train_reranker.py \
    --data_path ./train_data.jsonl \
    --generator_predictions ./generator_top_k_train.jsonl \
    --output_dir ./reranker_checkpoint \
    --epochs 5 --batch_size 32
```

**Time:** ~2-4 hours on GPU

### Step 3: Get Accuracy with Reranker

```bash
python run_inference_with_reranker.py \
    --data_path ./test_data.jsonl \
    --adapter_path ./lora-llama3-mapping \
    --reranker_checkpoint ./reranker_checkpoint/best_model.pt \
    --output_file ./predictions_reranked.jsonl \
    --eval_output ./evaluation_reranked.json
```

**Output files:**
- `predictions_reranked.jsonl` - Predictions
- `evaluation_reranked.json` - Accuracy metrics

**Console output shows:**
```
Exact Match: 35.7%
Partial Match: 65.3
Correct Keys: 2847/4235
```

---

## How Reranker Training Works

### Training Process

**Input:** Training data + generator's top-k predictions

**What happens:**

1. **Create contrastive pairs** for each placeholder:
   ```python
   Context: "int <ID_1> = 0; for (...) { <ID_1>++; }"
   Positive: "count"  # Gold name from training data
   Negative: "file"   # Wrong name (from generator errors or random)
   ```

2. **Train dual-encoder model:**
   - Context encoder: Encodes code around placeholder (CodeBERT)
   - Name encoder: Encodes candidate identifier (CodeBERT)
   - Goal: Make similarity(context, correct_name) > similarity(context, wrong_name)

3. **Loss function (InfoNCE):**
   ```
   Loss = -log(exp(score_correct) / (exp(score_correct) + exp(score_wrong)))
   ```
   Minimize this → model learns to score correct names higher

4. **Save checkpoints:**
   - Best model saved to `./reranker_checkpoint/best_model.pt`

### Negative Sampling

Where do wrong names come from?

1. **Generator errors** (best): Incorrect predictions from top-k
2. **Other variables**: Names from other placeholders in same function
3. **Random names**: Names from other functions in dataset

---

## How Reranker Inference Works

### Inference Process

**Input:** Test code + trained generator + trained reranker

**What happens:**

1. **Generator produces top-10 candidates** per placeholder:
   ```python
   Code: "int <ID_1> = 0; for (...) { <ID_1>++; }"
   
   Generator output:
   {
     "<ID_1>": ["count", "i", "index", "n", "total", 
                "num", "counter", "iter", "idx", "k"]
   }
   ```

2. **For each placeholder, reranker:**
   - Extracts local context (100 chars before/after)
   - Encodes context → 768-dim vector
   - Encodes all 10 candidates → 10 × 768-dim vectors
   - Computes scores: `cos_similarity(context, candidate)`
   - Applies penalties:
     - Collision penalty if name already used in function
     - Length penalty (slight preference for shorter names)

3. **Selects best candidate:**
   ```python
   Scores: [0.85, 0.72, 0.70, 0.68, 0.65, ...]
           ^^^^
           Best → "count"
   ```

---

## File Descriptions

| File | What it does |
|------|--------------|
| `generate_topk.py` | Generate top-10 candidates from fine-tuned generator |
| `train_reranker.py` | Train reranker with contrastive loss |
| `run_inference_with_reranker.py` | Run inference + get accuracy |
| `reranker.py` | Core reranker model (dual-encoder) |

---

## Expected Results

| System | Exact Match | Improvement |
|--------|-------------|-------------|
| Zero-shot | 6.1% | Baseline |
| Few-shot | 10.4% | +4.3pp |
| Fine-tuned Gen | 25-35% | - |
| **Gen + Rerank** | **30-40%** | **+5-10pp** |

---

## Troubleshooting

**Out of memory during training:**
```bash
python train_reranker.py ... --batch_size 16  # or 8
```

**Generator produces poor candidates:**
```bash
python generate_topk.py ... --num_samples 30 --temperature 0.9
```

**Want to test on small subset first:**
```bash
python run_inference_with_reranker.py ... --max_samples 100
```

---

## Key Hyperparameters

### Generator (Top-K)
- `k=10`: Number of candidates (5-15 recommended)
- `num_samples=20`: Diversity samples (more = better variety)
- `temperature=0.8`: Higher = more diverse (0.7-0.9)

### Reranker Training
- `epochs=5`: Training epochs (3-10 typical)
- `batch_size=32`: Adjust based on GPU memory
- `lr=1e-4`: Learning rate

### Reranker Inference
- `context_window=100`: Chars around placeholder
- `collision_penalty=-5.0`: Penalty for duplicate names

---

## Summary

1. **Fine-tune generator** → LoRA adapter
2. **Generate top-k** + **Train reranker** → Reranker checkpoint
3. **Run inference with reranker** → Get accuracy

Script for accuracy: `run_inference_with_reranker.py`

Expected: +5-10% improvement over generator-only

