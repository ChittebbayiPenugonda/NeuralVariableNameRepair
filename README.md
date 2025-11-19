# NeuralVariableNameRepair

Writing clean code is not just about making programs work; it is also about giving variables names that clearly communicate their purpose. This project explores training Llama 3.1 8B to understand and recover missing variable names based on the context of functions in C++.

## Setup

1. **Create `.env` file with your Hugging Face token:**
```bash
echo "HF_TOKEN=hf_your_token_here" > .env
```

2. **Create and activate conda environment:**
```bash
conda env create -f environment.yml
conda activate stack-cpp-extract
```

3. **Run the extraction script:**
```bash
python extract_functions.py
```

Output is saved to `data/data_cpp.jsonl` (sample in `example_output.jsonl`).

## Testing

```bash
# Test with 2 samples
python run_inference.py --data example_output.jsonl --max-samples 2

# Check outputs
cat llm_responses.jsonl
cat evaluation_results.json
```

**Note:** First run downloads Llama 3.1 8B (~16GB). Requires GPU with sufficient VRAM.

## Usage

### Run Inference with vLLM

```bash
# Run on sample data (5 examples)
python run_inference.py --data example_output.jsonl

# Run on full dataset
python run_inference.py --data data/data_cpp.jsonl

# Customize parameters
python run_inference.py \
    --data data/data_cpp.jsonl \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --max-samples 100 \
    --temperature 0.0 \
    --tensor-parallel-size 1
```

### Programmatic Usage

```python
from data_pipeline import DataPipeline
from evaluate_predictions import PredictionEvaluator

# Load data
pipeline = DataPipeline('example_output.jsonl')
pipeline.load_data()
input_texts, target_texts = pipeline.get_separate_arrays()

# Get LLM responses (see run_inference.py for full implementation)
# llm_responses = [llm(text) for text in input_texts]

# Evaluate
evaluator = PredictionEvaluator()
cleaned = evaluator.prepare_llm_responses(llm_responses)
metrics = evaluator.evaluate(cleaned, target_texts)
evaluator.print_report(metrics)
```

### Data Format

Each JSONL entry contains:
- `input_text`: Code with masked variables (`<ID_1>`, `<ID_2>`, etc.)
- `target_text`: JSON mapping of placeholders to actual variable names

LLM should output JSON: `{"<ID_1>": "varName", "<ID_2>": "anotherVar"}`

### Evaluation

The evaluator computes three metrics:
- **Exact Match**: Percentage of functions where all placeholders are correctly named
- **Partial Match**: Average similarity score (0-100) across all placeholders
- **Top-k Hit**: Percentage where gold name appears in top-k candidates

## Reranker: Two-Stage Inference for Better Accuracy

The reranker improves predictions by generating multiple candidates and selecting the best based on context.

### Complete Workflow (3 Steps)

#### Step 1: Fine-tune Generator (if not done yet)
```bash
python finetune_lora_llama.py \
    --data_path ./train_data.jsonl \
    --base_model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --out_dir ./lora-llama3-mapping \
    --epochs 3 --lr 2e-4 --batch_size 1 --grad_accum 8
```

#### Step 2: Train Reranker

First, generate top-k candidates from the generator:
```bash
python generate_topk.py \
    --data_path ./train_data.jsonl \
    --adapter_path ./lora-llama3-mapping \
    --output_file ./generator_top_k_train.jsonl \
    --k 10 --num_samples 20
```

Then train the reranker:
```bash
python train_reranker.py \
    --data_path ./train_data.jsonl \
    --generator_predictions ./generator_top_k_train.jsonl \
    --output_dir ./reranker_checkpoint \
    --epochs 5 --batch_size 32 --lr 1e-4
```

**What this does:**
- Creates training pairs: (code_context, correct_name, wrong_name)
- Trains dual-encoder with contrastive loss to distinguish good/bad names
- Saves best model to `./reranker_checkpoint/best_model.pt`
- Training time: ~2-4 hours on single GPU

#### Step 3: Get Accuracy with Reranker

Run inference and get evaluation metrics:
```bash
python run_inference_with_reranker.py \
    --data_path ./test_data.jsonl \
    --adapter_path ./lora-llama3-mapping \
    --reranker_checkpoint ./reranker_checkpoint/best_model.pt \
    --output_file ./predictions_reranked.jsonl \
    --eval_output ./evaluation_reranked.json \
    --k 10 --num_samples 20
```

**Output:**
- `predictions_reranked.jsonl`: Predicted variable names
- `evaluation_reranked.json`: Metrics (Exact Match, Partial Match, etc.)

### How It Works

**Two-Stage Pipeline:**

```
Stage 1 (Generator): Generate top-10 candidates per placeholder
  Code → Fine-tuned Llama → ["count", "i", "index", "n", "total", ...]

Stage 2 (Reranker): Score and select best candidate
  Code context + Candidates → Dual-encoder → Best pick: "count"
```

**Reranker Training (Contrastive Learning):**
1. Extract context around each placeholder (e.g., "int <ID_1> = 0; ... <ID_1>++;")
2. Create positive pairs: (context, correct_name) → Score should be HIGH
3. Create negative pairs: (context, wrong_name) → Score should be LOW
4. Train with InfoNCE loss to maximize positive similarity, minimize negative similarity

**Reranker Inference:**
1. Generator produces top-10 candidates
2. Reranker scores each based on local code context
3. Applies collision penalty (prevents duplicate names)
4. Selects highest-scoring candidate

**Expected improvement:** +5-10% Exact Match over generator-only 