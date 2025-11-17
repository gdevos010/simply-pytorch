"""Language Model Benchmark for Simply PyTorch Optimizers.

Trains a 125M parameter GPT-2 model on WikiText-103 to validate Cautious Weight Decay
across all 5 optimizers (SGD, Adam, AdamAtan2, Lion, Muon) with/without CWD.
"""

import gc
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math

from dataclasses import dataclass
from typing import Any

import lightning as L
import torch
import torch.nn.functional as F

from datasets import load_dataset
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2TokenizerFast

from simply_pytorch import SGD, Adam, AdamAtan2, Lion, Muon


@dataclass
class GPT2Config:
    """Configuration for GPT-2 model."""

    vocab_size: int = 50257  # GPT-2 vocabulary size
    n_layer: int = 12  # Number of transformer layers
    n_head: int = 12  # Number of attention heads
    n_embd: int = 768  # Embedding dimension
    block_size: int = 512  # Maximum sequence length
    dropout: float = 0.1  # Dropout probability
    bias: bool = True  # Use bias in linear layers

    @property
    def n_params(self) -> int:
        """Calculate approximate number of parameters."""
        # Embedding: vocab_size * n_embd + block_size * n_embd
        embedding = self.vocab_size * self.n_embd + self.block_size * self.n_embd
        # Each layer: 4 * n_embd^2 (attn) + 8 * n_embd^2 (mlp) = 12 * n_embd^2
        transformer = self.n_layer * 12 * self.n_embd**2
        # Output: n_embd * vocab_size
        output = self.n_embd * self.vocab_size
        return embedding + transformer + output


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Key, query, value projections for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # Causal mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch, sequence length, embedding dimension

        # Calculate query, key, values for all heads and split
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Causal self-attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Feed-forward network."""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x, approximate="tanh")
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP."""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2Model(nn.Module):
    """GPT-2 language model."""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList(
                    [TransformerBlock(config) for _ in range(config.n_layer)]
                ),
                "ln_f": nn.LayerNorm(config.n_embd),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, max is {self.config.block_size}"
        )
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # Forward through embeddings
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Forward through transformer blocks
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def count_parameters(self) -> int:
        """Count actual number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class WikiTextDataset(Dataset):
    """WikiText-103 dataset for language modeling."""

    def __init__(
        self, split: str, tokenizer: GPT2TokenizerFast, block_size: int = 512
    ) -> None:
        self.tokenizer = tokenizer
        self.block_size = block_size

        # Load dataset
        dataset = load_dataset(
            "wikitext", "wikitext-103-raw-v1", split=split, trust_remote_code=True
        )

        # Tokenize and concatenate
        print(f"Tokenizing {split} split...")
        all_tokens = []
        for example in dataset:
            text = example["text"]
            if text.strip():  # Skip empty lines
                tokens = tokenizer.encode(text)
                all_tokens.extend(tokens)

        # Create chunks
        self.examples = []
        for i in range(0, len(all_tokens) - block_size, block_size):
            self.examples.append(all_tokens[i : i + block_size + 1])

        print(f"Created {len(self.examples)} examples for {split}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tokens = self.examples[idx]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return {"input_ids": x, "labels": y}


class WikiTextDataModule(L.LightningDataModule):
    """Lightning data module for WikiText-103."""

    def __init__(
        self,
        batch_size: int = 32,
        block_size: int = 512,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.block_size = block_size
        self.num_workers = num_workers
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = WikiTextDataset(
                "train", self.tokenizer, self.block_size
            )
            self.val_dataset = WikiTextDataset(
                "validation", self.tokenizer, self.block_size
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class LMBenchmarkModule(L.LightningModule):
    """Lightning module for language model benchmark."""

    def __init__(
        self,
        config: GPT2Config,
        optimizer_name: str,
        use_cautious_wd: bool,
        learning_rate: float,
        weight_decay: float,
        warmup_steps: int = 2000,
        max_steps: int = 50000,
        **optimizer_kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = GPT2Model(config)
        self.optimizer_name = optimizer_name
        self.use_cautious_wd = use_cautious_wd
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.optimizer_kwargs = optimizer_kwargs

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.model(idx)

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        logits = self(batch["input_ids"])
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), batch["labels"].view(-1)
        )
        perplexity = torch.exp(loss)

        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        self.log("train/perplexity", perplexity, prog_bar=True, sync_dist=True)

        # Only log LR if trainer is attached (check internal _trainer attribute)
        if hasattr(self, "_trainer") and self._trainer is not None:
            self.log("train/lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)

        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        logits = self(batch["input_ids"])
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), batch["labels"].view(-1)
        )
        perplexity = torch.exp(loss)

        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/perplexity", perplexity, prog_bar=True, sync_dist=True)

        return loss

    def configure_optimizers(self) -> dict:
        # Create optimizer based on name
        optimizer_classes = {
            "sgd": SGD,
            "adam": Adam,
            "adamatan2": AdamAtan2,
            "lion": Lion,
            "muon": Muon,
        }

        optimizer_class = optimizer_classes[self.optimizer_name.lower()]

        # Build optimizer kwargs
        opt_kwargs = {
            "lr": self.learning_rate,
            "weight_decay": self.weight_decay,
            "use_cautious_wd": self.use_cautious_wd,
        }
        opt_kwargs.update(self.optimizer_kwargs)

        optimizer = optimizer_class(self.parameters(), **opt_kwargs)

        # Cosine learning rate scheduler with warmup
        def lr_lambda(current_step: int) -> float:
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            progress = float(current_step - self.warmup_steps) / float(
                max(1, self.max_steps - self.warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


def get_optimizer_config(optimizer_name: str, use_cautious_wd: bool) -> dict[str, Any]:
    """Get optimizer configuration based on name and CWD setting."""
    configs = {
        "adam": {
            "learning_rate": 1e-3,
            "weight_decay": 1e-3,
            "betas": (0.9, 0.95) if use_cautious_wd else (0.9, 0.999),
        },
        "adamatan2": {
            "learning_rate": 1e-4,
            "weight_decay": 1e-3,
            "betas": (0.9, 0.95) if use_cautious_wd else (0.9, 0.999),
        },
        "lion": {
            "learning_rate": 1e-4,
            "weight_decay": 0.1,
            "betas": (0.9, 0.95) if use_cautious_wd else (0.95, 0.98),
        },
        "muon": {
            "learning_rate": 0.02,
            "weight_decay": 0.01,
            "momentum": 0.95,
            "nesterov": True,
            "ns_steps": 5,
            "adam_betas": (0.9, 0.95),
        },
        "sgd": {
            "learning_rate": 0.1,
            "weight_decay": 1e-4,
        },
    }
    return configs[optimizer_name.lower()]


def train_single_config(
    optimizer_name: str,
    use_cautious_wd: bool,
    max_steps: int = 50000,
    val_check_interval: int = 1000,
    precision: str = "bf16-mixed",
) -> None:
    """Train a single optimizer configuration."""
    # Setup
    config = GPT2Config()
    opt_config = get_optimizer_config(optimizer_name, use_cautious_wd)

    # Create run name
    cwd_suffix = "_cwd" if use_cautious_wd else ""
    run_name = f"{optimizer_name}{cwd_suffix}"

    print(f"\n{'=' * 60}")
    print(f"Training: {run_name}")
    print(f"Model parameters: {config.n_params / 1e6:.1f}M")
    print(f"Optimizer config: {opt_config}")
    print(f"{'=' * 60}\n")

    # Data module
    data_module = WikiTextDataModule(batch_size=96, block_size=config.block_size)

    # Model
    learning_rate = opt_config.pop("learning_rate")
    weight_decay = opt_config.pop("weight_decay")
    model = LMBenchmarkModule(
        config=config,
        optimizer_name=optimizer_name,
        use_cautious_wd=use_cautious_wd,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=2000,
        max_steps=max_steps,
        **opt_config,
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{run_name}",
        filename="best",
        monitor="val/perplexity",
        mode="min",
        save_top_k=1,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Logger
    logger = WandbLogger(
        project="simply-pytorch-benchmark",
        name=run_name,
        save_dir="logs",
    )

    # Trainer
    trainer = L.Trainer(
        max_steps=max_steps,
        val_check_interval=val_check_interval,
        devices="auto",
        accelerator="gpu",
        strategy="ddp",
        precision=precision,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=100,
    )

    # Train
    trainer.fit(model, data_module)

    # Clean up to avoid GPU memory leaks between runs
    if isinstance(logger, WandbLogger):
        logger.experiment.finish()

    del trainer
    del model
    del data_module

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    """Run all 10 optimizer configurations."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optimizer",
        type=str,
        default="all",
        choices=["all", "sgd", "adam", "adamatan2", "lion", "muon"],
        help="Which optimizer to train",
    )
    parser.add_argument(
        "--max-steps", type=int, default=50000, help="Maximum training steps"
    )
    parser.add_argument(
        "--val-check-interval", type=int, default=100, help="Validation interval"
    )
    parser.add_argument(
        "--precision", type=str, default="bf16-mixed", help="Training precision"
    )
    args = parser.parse_args()

    optimizers = (
        ["adam", "adamatan2", "lion", "muon", "sgd"]
        if args.optimizer == "all"
        else [args.optimizer]
    )

    for optimizer_name in optimizers:
        for use_cautious_wd in [True, False]:
            train_single_config(
                optimizer_name=optimizer_name,
                use_cautious_wd=use_cautious_wd,
                max_steps=args.max_steps,
                val_check_interval=args.val_check_interval,
                precision=args.precision,
            )


if __name__ == "__main__":
    main()
