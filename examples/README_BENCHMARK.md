# Language Model Benchmark

Comprehensive benchmark for validating Cautious Weight Decay (CWD) across all Simply PyTorch optimizers on a 125M parameter GPT-2 model trained on WikiText-103.

## Setup

Install the benchmark dependencies:

```bash
pip install -e ".[dev]"
```

## Running the Benchmark

### Train All Optimizers

Train all 10 configurations (5 optimizers × 2 weight decay modes):

```bash
python examples/language_model_benchmark.py
```

### Train Specific Optimizer

Train only a specific optimizer:

```bash

# Train only Lion
python examples/language_model_benchmark.py --optimizer lion
```

### Configuration Options

```bash
python examples/language_model_benchmark.py \
    --optimizer all \                # all, sgd, adam, adamatan2, lion, muon
    --max-steps 50000 \              # Training steps
    --val-check-interval 100 \      # Validation frequency
    --precision bf16-mixed           # Training precision
```

## Model Architecture

- **Model**: GPT-2 (125M parameters)
- **Layers**: 12 transformer blocks
- **Hidden Size**: 768
- **Attention Heads**: 12
- **Sequence Length**: 512 tokens
- **Vocabulary Size**: 50,257 (GPT-2 tokenizer)

## Training Setup

- **Dataset**: WikiText-103
- **Training Steps**: ~50,000 (1 epoch)
- **Validation**: Every 100 steps
- **Precision**: bfloat16 mixed precision
- **Gradient Clipping**: 1.0
- **LR Schedule**: Cosine with 2,000 step warmup

## Analyzing Results

After training completes, analyze the results:

```bash
python examples/analyze_results.py --project simply-pytorch-benchmark
```

This generates:

- `results/loss_curves.png` - Training and validation loss curves
- `results/perplexity_comparison.png` - Final perplexity comparison
- `results/cwd_improvement.png` - CWD improvement percentages
- `results/summary_table.csv` - Detailed metrics table
- `results/cwd_improvements.csv` - CWD impact analysis

## Expected Results

Based on the CWD paper, you should observe:

1. **CWD Benefits**: Most optimizers show improved final validation perplexity with CWD
2. **Best with Adam-family**: Adam and AdamAtan2 benefit most from CWD
3. **Late Training**: CWD advantages become more apparent in later training stages
4. **Optimizer Rankings**: Typically Muon ≈ Lion > AdamAtan2 > Adam > SGD

## Monitoring Training

All runs are logged to WandB. View live progress at:

```
https://wandb.ai/<your-username>/simply-pytorch-benchmark
```
