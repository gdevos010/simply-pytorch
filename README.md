# Simply PyTorch: Optimizers with Cautious Weight Decay

PyTorch implementations of optimizers from the [Simply](https://github.com/google-deepmind/simply) codebase, including support for **Cautious Weight Decay (CWD)** as described in the paper ["Cautious Weight Decay"](https://arxiv.org/html/2510.12402v1).

## Features

- **Five Optimizers**: SGD, Adam, AdamAtan2, Lion, and Muon
- **Cautious Weight Decay**: Sign-selective weight decay that improves final loss
- **Drop-in Replacements**: Standard PyTorch `Optimizer` interface
- **No Extra Hyperparameters**: CWD uses the same λ as standard weight decay
- **Well-Tested**: Comprehensive test suite ported from JAX implementation

## What is Cautious Weight Decay?

Cautious Weight Decay (CWD) is a simple modification to standard weight decay that applies regularization **only when it makes sense**—when the optimizer update and parameter have the same sign (pointing in the same direction).

### The Problem with Standard Weight Decay

Standard decoupled weight decay applies regularization uniformly:

```python
x_t+1 = x_t - η_t(u_t + λ·x_t)
```

This can be counterproductive when the update `u_t` and parameter `x_t` point in opposite directions, as weight decay actively resists beneficial movement toward the optimum.

### The CWD Solution

Cautious Weight Decay applies decay only when signs align:

```python
x_t+1 = x_t - η_t(u_t + λ·𝕀(u_t·x_t ≥ 0)·x_t)
```

where `𝕀(u_t·x_t ≥ 0)` is an element-wise indicator function.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/simply-pytorch.git
cd simply-pytorch

# Install package
pip install -e .
```

## Quick Start

### Basic Usage

```python
import torch
import torch.nn as nn
from simply_pytorch import Adam

# Create model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Create optimizer with standard weight decay
optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

# Training loop
for data, target in dataloader:
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

### Using Cautious Weight Decay

Simply add `use_cautious_wd=True`:

```python
from simply_pytorch import Adam

# Adam with Cautious Weight Decay
optimizer = Adam(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.95),  # Paper recommends beta2=0.95 with CWD
    weight_decay=1e-3,
    use_cautious_wd=True  # Enable CWD!
)
```

## Optimizers

### 1. SGD

Simple gradient descent without momentum.

```python
from simply_pytorch import SGD

optimizer = SGD(
    model.parameters(),
    lr=0.1,
    weight_decay=1e-4,
    use_cautious_wd=False  # Enable for CWD
)
```

### 2. Adam

Adam optimizer with bias correction.

```python
from simply_pytorch import Adam

optimizer = Adam(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),  # (beta1, beta2)
    eps=1e-6,
    weight_decay=1e-3,
    use_cautious_wd=False
)
```

**Hyperparameter Recommendations:**

- Standard: `lr=1e-3`, `betas=(0.9, 0.999)`, `weight_decay=1e-3`
- With CWD: `lr=1e-3`, `betas=(0.9, 0.95)`, `weight_decay=1e-3`, `use_cautious_wd=True`

### 3. AdamAtan2

Adam optimizer with atan2-based updates for improved scale invariance.

```python
from simply_pytorch import AdamAtan2

optimizer = AdamAtan2(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),  # (beta1, beta2)
    weight_decay=1e-3,
    use_cautious_wd=False
)
```

**Key Features:**

- Uses `atan2(m, sqrt(v))` instead of `m / (sqrt(v) + eps)`
- No epsilon hyperparameter needed—naturally handles numerical stability
- Scale invariant: works across different parameter magnitudes
- From Google DeepMind paper "Scaling Exponents Across Parameterizations and Optimizers"

**Hyperparameter Recommendations:**

- Standard: `lr=1e-4`, `betas=(0.9, 0.999)`, `weight_decay=1e-3`
- With CWD: `lr=1e-4`, `betas=(0.9, 0.95)`, `weight_decay=1e-3`, `use_cautious_wd=True`

**When to Use:**

- Training with mixed parameter scales
- When epsilon tuning in Adam is problematic
- Seeking improved numerical stability without hyperparameter tuning

### 4. Lion

EvoLved Sign Momentum optimizer—memory efficient and powerful.

```python
from simply_pytorch import Lion

optimizer = Lion(
    model.parameters(),
    lr=1e-4,  # Typically 3-10x smaller than Adam
    betas=(0.95, 0.98),
    weight_decay=0.1,  # Can use higher values than Adam
    use_cautious_wd=False
)
```

**Key Features:**

- Sign-based updates: `update = sign(β₁·m + (1-β₁)·g)`
- Memory efficient: only stores first moment
- Often outperforms AdamW with proper tuning

**Hyperparameter Recommendations:**

- Standard: `lr=1e-4`, `betas=(0.95, 0.98)`, `weight_decay=0.1`
- With CWD: `lr=1e-4`, `betas=(0.9, 0.95)`, `weight_decay=0.1`, `use_cautious_wd=True`

### 5. Muon

Hybrid optimizer using Newton-Schulz orthogonalization for weights and Adam for biases.

```python
from simply_pytorch import Muon

optimizer = Muon(
    model.parameters(),
    lr=0.02,  # Paper suggests 0.02 for LLM pre-training
    momentum=0.95,
    nesterov=True,
    ns_steps=5,  # Newton-Schulz iterations
    adam_betas=(0.9, 0.95),
    weight_decay=0.01,
    use_cautious_wd=False,
    dim_threshold=10000  # Max dimension for Muon vs Adam
)
```

**Key Features:**

- Uses Newton-Schulz for 2D parameters (weights) with `max(dim) <= dim_threshold`
- Falls back to Adam for 1D parameters (biases, layer norms)
- Designed for transformer architectures

## Advanced Usage

### Multiple Parameter Groups

Apply different learning rates to different layers:

```python
from simply_pytorch import Adam

optimizer = Adam([
    {'params': model.encoder.parameters(), 'lr': 1e-3},
    {'params': model.decoder.parameters(), 'lr': 1e-4},
], weight_decay=1e-3, use_cautious_wd=True)
```

### Learning Rate Schedules

Compatible with PyTorch schedulers:

```python
from torch.optim.lr_scheduler import CosineAnnealingLR
from simply_pytorch import Lion

optimizer = Lion(model.parameters(), lr=1e-3, use_cautious_wd=True)
scheduler = CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(100):
    train(...)
    scheduler.step()
```


## Examples

Check out `examples/basic_usage.py` for comprehensive examples:

```bash
python examples/basic_usage.py
```

### Best Practices

1. Start with CWD enabled (`use_cautious_wd=True`)
2. Use paper's recommended hyperparameters (e.g., `beta2=0.95` for Adam+CWD)
3. Keep the same weight decay coefficient λ as your standard configuration
4. Monitor validation loss—CWD typically shows benefits late in training

## Citation

If you use Simply PyTorch or Cautious Weight Decay in your research, please cite:

```bibtex
@article{chen2025cautious,
  title={Cautious Weight Decay},
  author={Chen, Lizhang and Li, Jonathan and Liang, Kaizhao and Su, Baiyu and Xie, Cong and Wang, Nuo and Liang, Chen and Lao, Ni and Liu, Qiang},
  journal={arXiv preprint arXiv:2510.12402},
  year={2025}
}

@misc{liang2025simply,
  author={Chen Liang and Da Huang and Chengrun Yang and Xiaomeng Yang and Andrew Li and Xinchen Yan and Simply Contributors},
  title={{Simply: an experiment to accelerate and automate AI research}},
  year={2025},
  howpublished={GitHub repository},
  url={https://github.com/google-deepmind/simply}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

This PyTorch port is based on the JAX implementation from the [Simply](https://github.com/google-deepmind/simply) framework by Google DeepMind. The Cautious Weight Decay algorithm was developed by researchers at the University of Texas at Austin and Google.
