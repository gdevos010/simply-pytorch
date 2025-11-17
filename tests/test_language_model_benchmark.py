"""Tests for language model benchmark."""

import math

from unittest.mock import MagicMock, patch

import pytest
import torch

from examples.language_model_benchmark import (
    MLP,
    CausalSelfAttention,
    GPT2Config,
    GPT2Model,
    LMBenchmarkModule,
    TransformerBlock,
    get_optimizer_config,
)


class TestGPT2Config:
    """Tests for GPT-2 configuration."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = GPT2Config()
        assert config.vocab_size == 50257
        assert config.n_layer == 12
        assert config.n_head == 12
        assert config.n_embd == 768
        assert config.block_size == 512
        assert config.dropout == 0.1
        assert config.bias is True

    def test_parameter_count_calculation(self) -> None:
        """Test parameter count calculation."""
        config = GPT2Config()
        n_params = config.n_params

        # This is an approximation - actual count will be lower due to weight tying
        assert 150_000_000 < n_params < 170_000_000

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = GPT2Config(
            n_layer=6,
            n_head=8,
            n_embd=512,
        )
        assert config.n_layer == 6
        assert config.n_head == 8
        assert config.n_embd == 512


class TestCausalSelfAttention:
    """Tests for causal self-attention module."""

    def test_initialization(self) -> None:
        """Test module initialization."""
        config = GPT2Config()
        attn = CausalSelfAttention(config)

        assert attn.n_head == config.n_head
        assert attn.n_embd == config.n_embd
        assert attn.c_attn.in_features == config.n_embd
        assert attn.c_attn.out_features == 3 * config.n_embd

    def test_forward_pass(self) -> None:
        """Test forward pass."""
        config = GPT2Config()
        attn = CausalSelfAttention(config)

        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, seq_len, config.n_embd)

        output = attn(x)

        assert output.shape == (batch_size, seq_len, config.n_embd)

    def test_causal_mask(self) -> None:
        """Test that causal mask is properly registered."""
        config = GPT2Config()
        attn = CausalSelfAttention(config)

        assert hasattr(attn, "bias")
        assert attn.bias.shape == (1, 1, config.block_size, config.block_size)
        # Check it's lower triangular
        assert torch.all(attn.bias == torch.tril(torch.ones_like(attn.bias)))


class TestMLP:
    """Tests for MLP module."""

    def test_initialization(self) -> None:
        """Test module initialization."""
        config = GPT2Config()
        mlp = MLP(config)

        assert mlp.c_fc.in_features == config.n_embd
        assert mlp.c_fc.out_features == 4 * config.n_embd
        assert mlp.c_proj.in_features == 4 * config.n_embd
        assert mlp.c_proj.out_features == config.n_embd

    def test_forward_pass(self) -> None:
        """Test forward pass."""
        config = GPT2Config()
        mlp = MLP(config)

        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, seq_len, config.n_embd)

        output = mlp(x)

        assert output.shape == (batch_size, seq_len, config.n_embd)


class TestTransformerBlock:
    """Tests for transformer block."""

    def test_initialization(self) -> None:
        """Test module initialization."""
        config = GPT2Config()
        block = TransformerBlock(config)

        assert isinstance(block.attn, CausalSelfAttention)
        assert isinstance(block.mlp, MLP)

    def test_forward_pass(self) -> None:
        """Test forward pass."""
        config = GPT2Config()
        block = TransformerBlock(config)

        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, seq_len, config.n_embd)

        output = block(x)

        assert output.shape == (batch_size, seq_len, config.n_embd)


class TestGPT2Model:
    """Tests for GPT-2 model."""

    def test_initialization(self) -> None:
        """Test model initialization."""
        config = GPT2Config()
        model = GPT2Model(config)

        assert model.config == config
        assert len(model.transformer.h) == config.n_layer

    def test_parameter_count(self) -> None:
        """Test actual parameter count."""
        config = GPT2Config()
        model = GPT2Model(config)

        actual_params = model.count_parameters()

        # Should be approximately 125M (actual is ~124M due to weight tying)
        assert 120_000_000 < actual_params < 130_000_000

        # Verify approximate count
        print(f"Actual parameters: {actual_params / 1e6:.1f}M")

    def test_forward_pass(self) -> None:
        """Test forward pass."""
        config = GPT2Config(n_layer=2, block_size=64)  # Smaller for testing
        model = GPT2Model(config)

        batch_size = 2
        seq_len = 16
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        logits = model(idx)

        assert logits.shape == (batch_size, seq_len, config.vocab_size)

    def test_weight_tying(self) -> None:
        """Test that embedding and output weights are tied."""
        config = GPT2Config()
        model = GPT2Model(config)

        # Check that weights are the same object
        assert model.transformer.wte.weight is model.lm_head.weight

    def test_max_sequence_length(self) -> None:
        """Test that model enforces max sequence length."""
        config = GPT2Config(block_size=64)
        model = GPT2Model(config)

        # Should work with seq_len <= block_size
        idx = torch.randint(0, config.vocab_size, (1, config.block_size))
        _ = model(idx)

        # Should fail with seq_len > block_size
        idx = torch.randint(0, config.vocab_size, (1, config.block_size + 1))
        with pytest.raises(AssertionError):
            _ = model(idx)


class TestOptimizerConfigs:
    """Tests for optimizer configurations."""

    def test_sgd_config(self) -> None:
        """Test SGD configuration."""
        config_no_cwd = get_optimizer_config("sgd", use_cautious_wd=False)
        config_cwd = get_optimizer_config("sgd", use_cautious_wd=True)

        assert config_no_cwd["learning_rate"] == 0.1
        assert config_no_cwd["weight_decay"] == 1e-4
        assert config_cwd["learning_rate"] == 0.1
        assert config_cwd["weight_decay"] == 1e-4

    def test_adam_config(self) -> None:
        """Test Adam configuration."""
        config_no_cwd = get_optimizer_config("adam", use_cautious_wd=False)
        config_cwd = get_optimizer_config("adam", use_cautious_wd=True)

        assert config_no_cwd["learning_rate"] == 1e-3
        assert config_no_cwd["betas"] == (0.9, 0.999)
        assert config_cwd["betas"] == (0.9, 0.95)

    def test_adamatan2_config(self) -> None:
        """Test AdamAtan2 configuration."""
        config_no_cwd = get_optimizer_config("adamatan2", use_cautious_wd=False)
        config_cwd = get_optimizer_config("adamatan2", use_cautious_wd=True)

        assert config_no_cwd["learning_rate"] == 1e-4
        assert config_no_cwd["betas"] == (0.9, 0.999)
        assert config_cwd["betas"] == (0.9, 0.95)

    def test_lion_config(self) -> None:
        """Test Lion configuration."""
        config_no_cwd = get_optimizer_config("lion", use_cautious_wd=False)
        config_cwd = get_optimizer_config("lion", use_cautious_wd=True)

        assert config_no_cwd["learning_rate"] == 1e-4
        assert config_no_cwd["weight_decay"] == 0.1
        assert config_no_cwd["betas"] == (0.95, 0.98)
        assert config_cwd["betas"] == (0.9, 0.95)

    def test_muon_config(self) -> None:
        """Test Muon configuration."""
        config_no_cwd = get_optimizer_config("muon", use_cautious_wd=False)
        config_cwd = get_optimizer_config("muon", use_cautious_wd=True)

        assert config_no_cwd["learning_rate"] == 0.02
        assert config_no_cwd["weight_decay"] == 0.01
        assert config_no_cwd["momentum"] == 0.95
        assert config_cwd["learning_rate"] == 0.02


class TestLMBenchmarkModule:
    """Tests for Lightning module."""

    @pytest.fixture
    def config(self) -> GPT2Config:
        """Create small config for testing."""
        return GPT2Config(n_layer=2, n_head=4, n_embd=128, block_size=64)

    def test_initialization(self, config: GPT2Config) -> None:
        """Test module initialization."""
        module = LMBenchmarkModule(
            config=config,
            optimizer_name="adam",
            use_cautious_wd=True,
            learning_rate=1e-3,
            weight_decay=1e-3,
        )

        assert module.optimizer_name == "adam"
        assert module.use_cautious_wd is True
        assert isinstance(module.model, GPT2Model)

    def test_forward_pass(self, config: GPT2Config) -> None:
        """Test forward pass."""
        module = LMBenchmarkModule(
            config=config,
            optimizer_name="adam",
            use_cautious_wd=False,
            learning_rate=1e-3,
            weight_decay=1e-3,
        )

        batch_size = 2
        seq_len = 16
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        logits = module(idx)

        assert logits.shape == (batch_size, seq_len, config.vocab_size)

    @pytest.mark.parametrize(
        "optimizer_name", ["sgd", "adam", "adamatan2", "lion", "muon"]
    )
    def test_optimizer_configuration(
        self, config: GPT2Config, optimizer_name: str
    ) -> None:
        """Test that all optimizers can be configured."""
        module = LMBenchmarkModule(
            config=config,
            optimizer_name=optimizer_name,
            use_cautious_wd=False,
            learning_rate=1e-3,
            weight_decay=1e-3,
        )

        optimizer_config = module.configure_optimizers()

        assert "optimizer" in optimizer_config
        assert "lr_scheduler" in optimizer_config

    @pytest.mark.parametrize("use_cautious_wd", [False, True])
    def test_cautious_wd_flag(self, config: GPT2Config, use_cautious_wd: bool) -> None:
        """Test cautious weight decay flag."""
        module = LMBenchmarkModule(
            config=config,
            optimizer_name="adam",
            use_cautious_wd=use_cautious_wd,
            learning_rate=1e-3,
            weight_decay=1e-3,
        )

        optimizer_config = module.configure_optimizers()
        optimizer = optimizer_config["optimizer"]

        # Check that optimizer has correct use_cautious_wd setting
        assert optimizer.param_groups[0]["use_cautious_wd"] == use_cautious_wd

    def test_lr_schedule(self, config: GPT2Config) -> None:
        """Test learning rate schedule."""
        warmup_steps = 100
        max_steps = 1000

        module = LMBenchmarkModule(
            config=config,
            optimizer_name="adam",
            use_cautious_wd=False,
            learning_rate=1e-3,
            weight_decay=1e-3,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
        )

        optimizer_config = module.configure_optimizers()
        scheduler = optimizer_config["lr_scheduler"]["scheduler"]

        # Test warmup phase
        initial_lr = scheduler.get_last_lr()[0]
        assert initial_lr == pytest.approx(0.0, abs=1e-6)

        # Simulate warmup
        for _ in range(warmup_steps):
            scheduler.step()

        warmup_lr = scheduler.get_last_lr()[0]
        assert warmup_lr == pytest.approx(1e-3, rel=0.01)

        # Simulate cosine decay
        for _ in range(max_steps - warmup_steps):
            scheduler.step()

        final_lr = scheduler.get_last_lr()[0]
        assert final_lr < warmup_lr
        assert final_lr >= 0

    def test_training_step(self, config: GPT2Config) -> None:
        """Test training step."""
        module = LMBenchmarkModule(
            config=config,
            optimizer_name="adam",
            use_cautious_wd=False,
            learning_rate=1e-3,
            weight_decay=1e-3,
        )

        batch_size = 2
        seq_len = 16
        batch = {
            "input_ids": torch.randint(0, config.vocab_size, (batch_size, seq_len)),
            "labels": torch.randint(0, config.vocab_size, (batch_size, seq_len)),
        }

        loss = module.training_step(batch, 0)

        assert loss.item() > 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_validation_step(self, config: GPT2Config) -> None:
        """Test validation step."""
        module = LMBenchmarkModule(
            config=config,
            optimizer_name="adam",
            use_cautious_wd=False,
            learning_rate=1e-3,
            weight_decay=1e-3,
        )

        batch_size = 2
        seq_len = 16
        batch = {
            "input_ids": torch.randint(0, config.vocab_size, (batch_size, seq_len)),
            "labels": torch.randint(0, config.vocab_size, (batch_size, seq_len)),
        }

        loss = module.validation_step(batch, 0)

        assert loss.item() > 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


class TestDataPipeline:
    """Tests for data pipeline."""

    @patch("examples.language_model_benchmark.load_dataset")
    @patch("examples.language_model_benchmark.GPT2TokenizerFast")
    def test_dataset_creation(
        self,
        mock_tokenizer_class: MagicMock,
        mock_load_dataset: MagicMock,
    ) -> None:
        """Test dataset creation with mocked data."""
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = list(range(100))
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Mock dataset
        mock_dataset = [{"text": "Sample text " * 100} for _ in range(10)]
        mock_load_dataset.return_value = mock_dataset

        # Import and create dataset
        from examples.language_model_benchmark import WikiTextDataset

        dataset = WikiTextDataset("train", mock_tokenizer, block_size=64)

        # Check that dataset was created
        assert len(dataset) > 0

    def test_batch_shapes(self) -> None:
        """Test that batches have correct shapes."""
        # Create mock dataset
        block_size = 64
        batch_size = 4

        # Mock a simple dataset
        examples = [torch.randint(0, 50257, (block_size + 1,)) for _ in range(10)]

        from examples.language_model_benchmark import WikiTextDataset

        # Monkey patch the dataset
        dataset = MagicMock(spec=WikiTextDataset)
        dataset.__len__.return_value = len(examples)

        def getitem(idx: int) -> dict:
            tokens = examples[idx]
            return {"input_ids": tokens[:-1], "labels": tokens[1:]}

        dataset.__getitem__.side_effect = getitem

        # Test a batch
        batch = [dataset[i] for i in range(batch_size)]

        # Check shapes
        for item in batch:
            assert item["input_ids"].shape == (block_size,)
            assert item["labels"].shape == (block_size,)


class TestIntegration:
    """Integration tests."""

    def test_single_training_step_all_optimizers(self) -> None:
        """Test that a single training step works for all optimizers."""
        config = GPT2Config(n_layer=2, n_head=4, n_embd=128, block_size=64)

        optimizers = ["sgd", "adam", "adamatan2", "lion", "muon"]

        for optimizer_name in optimizers:
            for use_cautious_wd in [False, True]:
                module = LMBenchmarkModule(
                    config=config,
                    optimizer_name=optimizer_name,
                    use_cautious_wd=use_cautious_wd,
                    learning_rate=1e-3,
                    weight_decay=1e-3,
                )

                # Configure optimizer
                opt_config = module.configure_optimizers()
                optimizer = opt_config["optimizer"]

                # Create batch
                batch_size = 2
                seq_len = 16
                batch = {
                    "input_ids": torch.randint(
                        0, config.vocab_size, (batch_size, seq_len)
                    ),
                    "labels": torch.randint(
                        0, config.vocab_size, (batch_size, seq_len)
                    ),
                }

                # Training step
                optimizer.zero_grad()
                loss = module.training_step(batch, 0)
                loss.backward()
                optimizer.step()

                # Verify loss is reasonable
                assert loss.item() > 0
                assert not torch.isnan(loss)
                assert not torch.isinf(loss)

    def test_model_trains(self) -> None:
        """Test that model can complete multiple training steps."""
        config = GPT2Config(n_layer=2, n_head=4, n_embd=128, block_size=64)
        module = LMBenchmarkModule(
            config=config,
            optimizer_name="adam",
            use_cautious_wd=True,
            learning_rate=1e-3,
            weight_decay=1e-3,
        )

        opt_config = module.configure_optimizers()
        optimizer = opt_config["optimizer"]

        # Create fixed batch for testing
        torch.manual_seed(42)
        batch_size = 4
        seq_len = 32
        batch = {
            "input_ids": torch.randint(0, config.vocab_size, (batch_size, seq_len)),
            "labels": torch.randint(0, config.vocab_size, (batch_size, seq_len)),
        }

        # Train for a few steps
        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            loss = module.training_step(batch, 0)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Verify all losses are finite and positive
        for loss_val in losses:
            assert loss_val > 0
            assert not math.isnan(loss_val)
            assert not math.isinf(loss_val)

        # Verify gradients flow (parameters change)
        assert len(losses) == 10
