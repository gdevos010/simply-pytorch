"""Basic usage examples for Simply PyTorch optimizers.

This script demonstrates how to use each optimizer (SGD, Adam, Lion, Muon)
with both standard and Cautious Weight Decay (CWD).
"""

import torch

from torch import nn
from torch.nn.modules.container import Sequential

from simply_pytorch import SGD, Adam, Lion, Muon


def create_simple_model() -> Sequential:
    """Create a simple 2-layer neural network for demonstration."""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
    )


def example_sgd() -> None:
    """Example: Using SGD optimizer."""
    print("\n=== SGD Optimizer ===")

    model = create_simple_model()
    optimizer = SGD(model.parameters(), lr=0.1, weight_decay=1e-4)

    # Dummy forward pass
    x = torch.randn(32, 10)
    y = torch.randint(0, 5, (32,))

    output = model(x)
    loss = nn.functional.cross_entropy(output, y)

    print(f"Initial loss: {loss.item():.4f}")

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("SGD step completed!")


def example_sgd_with_cwd() -> None:
    """Example: Using SGD with Cautious Weight Decay."""
    print("\n=== SGD with Cautious Weight Decay ===")

    model = create_simple_model()
    optimizer = SGD(
        model.parameters(),
        lr=0.1,
        weight_decay=1e-4,
        use_cautious_wd=True,  # Enable CWD
    )

    # Dummy forward pass
    x = torch.randn(32, 10)
    y = torch.randint(0, 5, (32,))

    output = model(x)
    loss = nn.functional.cross_entropy(output, y)

    print(f"Initial loss: {loss.item():.4f}")

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("SGD with CWD step completed!")
    print("CWD only applies weight decay when update and param signs align")


def example_adam() -> None:
    """Example: Using Adam optimizer."""
    print("\n=== Adam Optimizer ===")

    model = create_simple_model()
    optimizer = Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-3)

    # Training loop
    for epoch in range(3):
        x = torch.randn(32, 10)
        y = torch.randint(0, 5, (32,))

        output = model(x)
        loss = nn.functional.cross_entropy(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")


def example_adam_with_cwd() -> None:
    """Example: Using Adam with Cautious Weight Decay."""
    print("\n=== Adam with Cautious Weight Decay ===")

    model = create_simple_model()
    optimizer = Adam(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.95),  # Paper recommends beta2=0.95 for CWD
        weight_decay=1e-3,
        use_cautious_wd=True,  # Enable CWD
    )

    # Training loop
    for epoch in range(3):
        x = torch.randn(32, 10)
        y = torch.randint(0, 5, (32,))

        output = model(x)
        loss = nn.functional.cross_entropy(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    print("Adam with CWD completed! Often achieves better generalization.")


def example_lion() -> None:
    """Example: Using Lion optimizer."""
    print("\n=== Lion Optimizer ===")

    model = create_simple_model()
    # Note: Lion typically uses smaller lr (3-10x smaller than Adam)
    # but can use higher weight decay
    optimizer = Lion(
        model.parameters(),
        lr=1e-4,  # Smaller than Adam
        betas=(0.95, 0.98),
        weight_decay=0.1,  # Higher than typical
    )

    # Training loop
    for epoch in range(3):
        x = torch.randn(32, 10)
        y = torch.randint(0, 5, (32,))

        output = model(x)
        loss = nn.functional.cross_entropy(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    print("Lion uses sign-based updates, very memory efficient!")


def example_lion_with_cwd() -> None:
    """Example: Using Lion with Cautious Weight Decay."""
    print("\n=== Lion with Cautious Weight Decay ===")

    model = create_simple_model()
    optimizer = Lion(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.95),  # CWD paper recommendations
        weight_decay=0.1,
        use_cautious_wd=True,
    )

    # Training loop
    for epoch in range(3):
        x = torch.randn(32, 10)
        y = torch.randint(0, 5, (32,))

        output = model(x)
        loss = nn.functional.cross_entropy(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")


def example_muon() -> None:
    """Example: Using Muon optimizer."""
    print("\n=== Muon Optimizer ===")

    model = create_simple_model()
    optimizer = Muon(
        model.parameters(),
        lr=0.02,  # Paper suggests 0.02 for LLM pre-training
        momentum=0.95,
        nesterov=True,
        weight_decay=0.01,
    )

    # Training loop
    for epoch in range(3):
        x = torch.randn(32, 10)
        y = torch.randint(0, 5, (32,))

        output = model(x)
        loss = nn.functional.cross_entropy(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    print("Muon uses Newton-Schulz for weights, Adam for biases/norms!")


def example_muon_with_cwd() -> None:
    """Example: Using Muon with Cautious Weight Decay."""
    print("\n=== Muon with Cautious Weight Decay ===")

    model = create_simple_model()
    optimizer = Muon(
        model.parameters(),
        lr=0.02,
        momentum=0.95,
        weight_decay=0.01,
        use_cautious_wd=True,
    )

    # Training loop
    for epoch in range(3):
        x = torch.randn(32, 10)
        y = torch.randint(0, 5, (32,))

        output = model(x)
        loss = nn.functional.cross_entropy(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")


def example_param_groups() -> None:
    """Example: Using different learning rates for different layers."""
    print("\n=== Parameter Groups Example ===")

    model = create_simple_model()

    # Different learning rates for different layers
    optimizer = Adam(
        [
            {"params": model[0].parameters(), "lr": 1e-3},  # First layer
            {"params": model[2].parameters(), "lr": 1e-4},  # Second layer
        ],
        weight_decay=1e-3,
    )

    # Training
    x = torch.randn(32, 10)
    y = torch.randint(0, 5, (32,))

    output = model(x)
    loss = nn.functional.cross_entropy(output, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Different learning rates applied to different layers!")


def compare_standard_vs_cwd() -> None:
    """Compare standard weight decay vs Cautious Weight Decay."""
    print("\n=== Comparing Standard WD vs CWD ===")

    # Standard weight decay
    model1 = create_simple_model()
    opt1 = Adam(model1.parameters(), lr=1e-3, weight_decay=1e-3, use_cautious_wd=False)

    # Cautious weight decay
    model2 = create_simple_model()
    opt2 = Adam(model2.parameters(), lr=1e-3, weight_decay=1e-3, use_cautious_wd=True)

    # Copy weights to ensure same starting point
    for p1, p2 in zip(model1.parameters(), model2.parameters(), strict=False):
        p2.data.copy_(p1.data)

    losses_standard = []
    losses_cwd = []

    # Training loop
    for _epoch in range(10):
        x = torch.randn(32, 10)
        y = torch.randint(0, 5, (32,))

        # Standard WD
        output1 = model1(x)
        loss1 = nn.functional.cross_entropy(output1, y)
        opt1.zero_grad()
        loss1.backward()
        opt1.step()
        losses_standard.append(loss1.item())

        # CWD
        output2 = model2(x)
        loss2 = nn.functional.cross_entropy(output2, y)
        opt2.zero_grad()
        loss2.backward()
        opt2.step()
        losses_cwd.append(loss2.item())

    print(f"\nFinal loss with standard WD: {losses_standard[-1]:.4f}")
    print(f"Final loss with CWD: {losses_cwd[-1]:.4f}")
    print(
        "\nCWD often achieves better final loss by being more selective about regularization!"
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Simply PyTorch Optimizers - Basic Usage Examples")
    print("=" * 60)

    # Run all examples
    example_sgd()
    example_sgd_with_cwd()
    example_adam()
    example_adam_with_cwd()
    example_lion()
    example_lion_with_cwd()
    example_muon()
    example_muon_with_cwd()
    example_param_groups()
    compare_standard_vs_cwd()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
