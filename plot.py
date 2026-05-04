"""
Plot results from train.py output.

Usage:
    python plot.py                        # reads results/*.json
    python plot.py --out results/my_run   # custom glob path
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "curriculum": "#2196F3",  # blue
    "random":     "#FF9800",  # orange
    "none":       "#9E9E9E",  # grey
}
LABELS = {
    "curriculum": "Curriculum generator (ours)",
    "random":     "Random curriculum",
    "none":       "No curriculum",
}


def load_results(results_dir: str) -> dict[str, list[dict]]:
    """Load all JSON result files, keyed by mode."""
    data: dict[str, list[dict]] = {}
    for path in sorted(Path(results_dir).glob("*.json")):
        mode = path.stem.rsplit("_", 1)[0]  # strip seed suffix
        with open(path) as f:
            history = json.load(f)
        # If multiple seeds exist for the same mode, keep only the last loaded
        data[mode] = history
    return data


def smooth(values: list[float], window: int = 5) -> np.ndarray:
    if len(values) <= window:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_success(data: dict[str, list[dict]], ax: plt.Axes) -> None:
    for mode, history in data.items():
        iters   = [r["iteration"] for r in history if "mean_success" in r]
        success = [r["mean_success"] for r in history if "mean_success" in r]
        color   = COLORS.get(mode, "black")
        label   = LABELS.get(mode, mode)

        ax.plot(iters, success, alpha=0.25, color=color, linewidth=0.8)
        smoothed = smooth(success)
        offset   = (len(success) - len(smoothed)) // 2
        ax.plot(
            iters[offset : offset + len(smoothed)],
            smoothed,
            color=color,
            linewidth=2.0,
            label=label,
        )

    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("Mean success rate on original tasks")
    ax.set_title("Curriculum generator vs baselines")
    ax.legend()
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)


def plot_generator_reward(data: dict[str, list[dict]], ax: plt.Axes) -> None:
    """Plot generator reward signal over time (curriculum mode only)."""
    if "curriculum" not in data:
        return
    history = data["curriculum"]
    iters   = [r["iteration"] for r in history if "generator_reward" in r]
    rewards = [r["generator_reward"] for r in history if "generator_reward" in r]
    if not iters:
        return

    ax.plot(iters, rewards, alpha=0.3, color=COLORS["curriculum"], linewidth=0.8)
    smoothed = smooth(rewards)
    offset   = (len(rewards) - len(smoothed)) // 2
    ax.plot(
        iters[offset : offset + len(smoothed)],
        smoothed,
        color=COLORS["curriculum"],
        linewidth=2.0,
    )
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("Generator reward (Δsuccess − penalty)")
    ax.set_title("Generator reward signal")
    ax.grid(alpha=0.3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results")
    parser.add_argument("--save", default=None, help="Save figure to this path")
    args = parser.parse_args()

    data = load_results(args.out)
    if not data:
        print(f"No result files found in {args.out}/")
        return

    has_gen_reward = "curriculum" in data and any(
        "generator_reward" in r for r in data["curriculum"]
    )
    ncols = 2 if has_gen_reward else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    plot_success(data, axes[0])
    if ncols == 2:
        plot_generator_reward(data, axes[1])

    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=150)
        print(f"Figure saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
