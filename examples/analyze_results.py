"""Analysis script for language model benchmark results.

Loads results from WandB and generates comparison plots and statistics.
"""

import argparse

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb


def load_wandb_runs(
    project: str = "simply-pytorch-benchmark",
) -> dict[str, pd.DataFrame]:
    """Load all runs from WandB project."""
    api = wandb.Api()
    runs = api.runs(project)

    run_data = {}
    print(f"Loading {len(runs)} runs from WandB project: {project}")

    for run in runs:
        run_name = run.name
        print(f"Loading: {run_name}")

        # Get history
        history = run.history()

        if len(history) > 0:
            run_data[run_name] = history
        else:
            print(f"  Warning: No data for {run_name}")

    return run_data


def extract_metrics(run_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Extract key metrics from all runs."""
    summary_data = []

    for run_name, df in run_data.items():
        # Parse optimizer name and CWD status
        if "_cwd" in run_name:
            optimizer = run_name.replace("_cwd", "")
            use_cwd = True
        else:
            optimizer = run_name
            use_cwd = False

        # Calculate metrics
        if "val/loss" in df.columns:
            final_val_loss = df["val/loss"].dropna().iloc[-1]
            final_val_ppl = df["val/perplexity"].dropna().iloc[-1]
            min_val_loss = df["val/loss"].dropna().min()
            min_val_ppl = df["val/perplexity"].dropna().min()

            summary_data.append(
                {
                    "optimizer": optimizer,
                    "use_cwd": use_cwd,
                    "final_val_loss": final_val_loss,
                    "final_val_perplexity": final_val_ppl,
                    "best_val_loss": min_val_loss,
                    "best_val_perplexity": min_val_ppl,
                }
            )

    return pd.DataFrame(summary_data)


def plot_loss_curves(run_data: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Plot training and validation loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Define colors for optimizers
    colors = {
        "sgd": "C0",
        "adam": "C1",
        "adamatan2": "C2",
        "lion": "C3",
        "muon": "C4",
    }

    # Plot training loss
    for run_name, df in run_data.items():
        if "train/loss" not in df.columns:
            continue

        optimizer = run_name.replace("_cwd", "")
        use_cwd = "_cwd" in run_name
        color = colors.get(optimizer, "black")
        linestyle = "-" if use_cwd else "--"
        label = f"{optimizer.upper()}{' (CWD)' if use_cwd else ''}"

        # Smooth the curve
        window = 50
        train_loss = df["train/loss"].dropna()
        if len(train_loss) > window:
            train_loss_smooth = train_loss.rolling(window=window).mean()
            ax1.plot(
                train_loss_smooth,
                label=label,
                color=color,
                linestyle=linestyle,
                alpha=0.8,
            )

    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss Curves")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Plot validation loss
    for run_name, df in run_data.items():
        if "val/loss" not in df.columns:
            continue

        optimizer = run_name.replace("_cwd", "")
        use_cwd = "_cwd" in run_name
        color = colors.get(optimizer, "black")
        linestyle = "-" if use_cwd else "--"
        label = f"{optimizer.upper()}{' (CWD)' if use_cwd else ''}"

        val_loss = df["val/loss"].dropna()
        # Get corresponding step numbers
        steps = (
            df.loc[val_loss.index, "_step"] if "_step" in df.columns else val_loss.index
        )
        ax2.plot(
            steps,
            val_loss,
            label=label,
            color=color,
            linestyle=linestyle,
            alpha=0.8,
            marker="o",
            markersize=3,
        )

    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Validation Loss")
    ax2.set_title("Validation Loss Curves")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "loss_curves.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir / 'loss_curves.png'}")
    plt.close()


def plot_perplexity_comparison(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot perplexity comparison bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare data for grouped bar chart
    optimizers = summary_df["optimizer"].unique()
    x = np.arange(len(optimizers))
    width = 0.35

    # Get final perplexities
    no_cwd = []
    with_cwd = []

    for opt in optimizers:
        no_cwd_val = summary_df[
            (summary_df["optimizer"] == opt) & (~summary_df["use_cwd"])
        ]["final_val_perplexity"].values
        with_cwd_val = summary_df[
            (summary_df["optimizer"] == opt) & (summary_df["use_cwd"])
        ]["final_val_perplexity"].values

        no_cwd.append(no_cwd_val[0] if len(no_cwd_val) > 0 else 0)
        with_cwd.append(with_cwd_val[0] if len(with_cwd_val) > 0 else 0)

    ax.bar(x - width / 2, no_cwd, width, label="Standard WD", alpha=0.8)
    ax.bar(x + width / 2, with_cwd, width, label="Cautious WD", alpha=0.8)

    ax.set_xlabel("Optimizer")
    ax.set_ylabel("Final Validation Perplexity")
    ax.set_title("Final Validation Perplexity Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([opt.upper() for opt in optimizers])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for i, (nc, wc) in enumerate(zip(no_cwd, with_cwd, strict=False)):
        if nc > 0:
            ax.text(
                i - width / 2, nc, f"{nc:.2f}", ha="center", va="bottom", fontsize=8
            )
        if wc > 0:
            ax.text(
                i + width / 2, wc, f"{wc:.2f}", ha="center", va="bottom", fontsize=8
            )

    plt.tight_layout()
    plt.savefig(output_dir / "perplexity_comparison.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir / 'perplexity_comparison.png'}")
    plt.close()


def plot_cwd_improvement(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot CWD improvement percentage for each optimizer."""
    fig, ax = plt.subplots(figsize=(10, 6))

    improvements = []
    optimizers = []

    for opt in summary_df["optimizer"].unique():
        no_cwd = summary_df[
            (summary_df["optimizer"] == opt) & (~summary_df["use_cwd"])
        ]["final_val_perplexity"].values
        with_cwd = summary_df[
            (summary_df["optimizer"] == opt) & (summary_df["use_cwd"])
        ]["final_val_perplexity"].values

        if len(no_cwd) > 0 and len(with_cwd) > 0:
            improvement = ((no_cwd[0] - with_cwd[0]) / no_cwd[0]) * 100
            improvements.append(improvement)
            optimizers.append(opt.upper())

    colors = ["green" if imp > 0 else "red" for imp in improvements]
    ax.barh(optimizers, improvements, color=colors, alpha=0.8)
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Perplexity Improvement (%)")
    ax.set_ylabel("Optimizer")
    ax.set_title("CWD Improvement Over Standard Weight Decay")
    ax.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for i, imp in enumerate(improvements):
        ax.text(imp, i, f"{imp:+.2f}%", ha="left" if imp > 0 else "right", va="center")

    plt.tight_layout()
    plt.savefig(output_dir / "cwd_improvement.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir / 'cwd_improvement.png'}")
    plt.close()


def create_summary_table(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """Create summary table of results."""
    # Pivot table
    pivot_df = summary_df.pivot_table(
        index="optimizer",
        columns="use_cwd",
        values=["final_val_loss", "final_val_perplexity", "best_val_perplexity"],
    )

    # Format and save
    pivot_df = pivot_df.round(4)
    pivot_df.to_csv(output_dir / "summary_table.csv")
    print("\nSummary Table:")
    print(pivot_df)
    print(f"\nSaved: {output_dir / 'summary_table.csv'}")

    # Calculate improvements
    improvement_data = []
    for opt in summary_df["optimizer"].unique():
        no_cwd = summary_df[(summary_df["optimizer"] == opt) & (~summary_df["use_cwd"])]
        with_cwd = summary_df[
            (summary_df["optimizer"] == opt) & (summary_df["use_cwd"])
        ]

        if len(no_cwd) > 0 and len(with_cwd) > 0:
            ppl_improvement = (
                (
                    no_cwd["final_val_perplexity"].values[0]
                    - with_cwd["final_val_perplexity"].values[0]
                )
                / no_cwd["final_val_perplexity"].values[0]
            ) * 100

            improvement_data.append(
                {
                    "optimizer": opt.upper(),
                    "standard_ppl": no_cwd["final_val_perplexity"].values[0],
                    "cwd_ppl": with_cwd["final_val_perplexity"].values[0],
                    "improvement_%": ppl_improvement,
                }
            )

    improvement_df = pd.DataFrame(improvement_data)
    improvement_df = improvement_df.sort_values("improvement_%", ascending=False)
    improvement_df.to_csv(output_dir / "cwd_improvements.csv", index=False)
    print("\nCWD Improvements:")
    print(improvement_df)
    print(f"\nSaved: {output_dir / 'cwd_improvements.csv'}")


def main() -> None:
    """Main analysis function."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project",
        type=str,
        default="simply-pytorch-benchmark",
        help="WandB project name",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for plots and tables",
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Simply PyTorch Benchmark Analysis")
    print(f"{'=' * 60}\n")

    # Load data
    run_data = load_wandb_runs(args.project)

    if not run_data:
        print("No runs found in WandB project!")
        return

    # Extract metrics
    summary_df = extract_metrics(run_data)

    # Generate plots
    print(f"\n{'=' * 60}")
    print("Generating plots...")
    print(f"{'=' * 60}\n")

    plot_loss_curves(run_data, output_dir)
    plot_perplexity_comparison(summary_df, output_dir)
    plot_cwd_improvement(summary_df, output_dir)

    # Create tables
    print(f"\n{'=' * 60}")
    print("Creating summary tables...")
    print(f"{'=' * 60}\n")

    create_summary_table(summary_df, output_dir)

    print(f"\n{'=' * 60}")
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
