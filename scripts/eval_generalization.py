#!/usr/bin/env python3
"""eval_checkpoints_neurips.py
====================================================
Evaluate a series of *open‑r1* RL‑fine‑tuned checkpoints on a set of
in‑distribution and out‑of‑distribution tasks, log the results,
produce NeurIPS‑quality plots and correlation statistics – **while
re‑using the existing open‑r1 code‑base** so we avoid duplication and
stay 100 % consistent with the training pipeline.

Usage
-----
```bash
python eval_checkpoints_neurips.py \
  --exp-dir /mnt/pdata/mr971/gsm8k_only \
  --save-dir /mnt/pdata/mr971/analysis/gsm8k_only \
  --device cuda:0 --batch-size 8 --max-new-tokens 32
```
The script auto‑discovers every `checkpoint-*` folder under
`--exp-dir`, evaluates each on all tasks, and writes:

* `metrics.csv` – tidy accuracy table *(checkpoint × task)*.
* `learning_curve.pdf` – accuracy vs. checkpoint for every task.
* `task_correlation_heatmap.pdf` – Pearson *r* between tasks.
* `scatter_matrix.pdf` – pair‑wise scatter plots (accuracy).

All figures are vector (PDF, 300 dpi) and use a compact NeurIPS style.
"""
from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import torch
from datasets import load_dataset, Dataset
from tqdm import tqdm

# ── W&B additions ───────────────────────────────────────────────────────
import os
import wandb
# -----------------------------------------------------------------------

# ── open‑r1 imports (reuse!) ────────────────────────────────────────────
from trl import ModelConfig
from open_r1.configs import SFTConfig # Need a dummy training config
from open_r1.utils import get_model, get_tokenizer
from open_r1.rewards import accuracy_reward  # robust LaTeX / boxed parser

# ----------------------------------------------------------------------
#  Tasks – *one* central dict you can tweak for more OOD evaluations
# ----------------------------------------------------------------------

TASKS: dict[str, tuple[str, str, str]] = {
    # in-distribution
    "gsm8k"            : ("src/open_r1/data/combined_loader.py", "gsm8k",             "train"),

    # out-of-distribution
    "advent_of_code"   : ("src/open_r1/data/combined_loader.py", "advent_of_code",    "train"),
    "bespoke_stratos"  : ("src/open_r1/data/combined_loader.py", "bespoke_stratos",   "train"),
    "brainteaser"      : ("src/open_r1/data/combined_loader.py", "brainteaser",       "train[:1000]"),
    "connections"      : ("src/open_r1/data/combined_loader.py", "connections",       "train"),
    "countdown"        : ("src/open_r1/data/combined_loader.py", "countdown",         "train"),
    "humaneval"        : ("src/open_r1/data/combined_loader.py", "humaneval",         "test"),
    "logiqa"           : ("src/open_r1/data/combined_loader.py", "logiqa",            "train"),
    "macgyver"         : ("src/open_r1/data/combined_loader.py", "macgyver",          "train"),
    "math"             : ("src/open_r1/data/combined_loader.py", "math",              "test"),
    "multiply"         : ("src/open_r1/data/combined_loader.py", "multiply",          "train[:1000]"),
    "numina_math_tir"  : ("src/open_r1/data/combined_loader.py", "numina_math_tir",   "train"),
    "project_euler"    : ("src/open_r1/data/combined_loader.py", "project_euler",     "train"),
    "riddlesense"      : ("src/open_r1/data/combined_loader.py", "riddlesense",       "validation"),
    "splat"            : ("src/open_r1/data/combined_loader.py", "splat",             "train"),
}


# ----------------------------------------------------------------------
#  Helper: aesthetic Matplotlib defaults (NeurIPS‑like)
# ----------------------------------------------------------------------

def set_neurips_style():
    plt.rcParams.update({
        "font.size": 9,
        "font.family": "sans-serif",
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.5,
        "pdf.fonttype": 42,  # editable text in Illustrator
    })

# ----------------------------------------------------------------------
#  Accuracy evaluation re‑using open‑r1 reward helpers
# ----------------------------------------------------------------------

def _batchify(dataset: Dataset, batch_size: int):
    """Simple generator that yields slices of the 🤗 Dataset."""
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]


def accuracy_for_dataset(model, tokenizer, dataset: Dataset, *, device: torch.device, batch_size: int, max_new_tokens: int) -> float:
    """Compute accuracy via open‑r1 `accuracy_reward`, which parses LaTeX, boxed
    answers, units, etc. Any example returning `None` is skipped. The model is
    *expected* to answer after the `<answer>` marker, but we do *not* enforce
    any additional formatting – we trust the reward parser.
    """
    model.eval()
    n_correct = 0
    n_total = 0

    with torch.no_grad():
        for batch in _batchify(dataset, batch_size):
            # Correctly access prompts and solutions from the batch dictionary
            num_examples_in_batch = len(batch["prompt"])  # Get batch size from a column
            prompts = [batch["prompt"][i] + "\n<answer>\n" for i in range(num_examples_in_batch)]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
            # Use do_sample=False for greedy decoding when temperature is effectively 0
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=None, do_sample=False)
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Completions need to be wrapped correctly for the reward function
            # Extract the generated text *after* the prompt+answer marker
            generated_texts = []
            for i in range(num_examples_in_batch):
                # Find the end of the input prompt within the decoded output
                prompt_end_index = decoded[i].find("\n<answer>\n")
                if prompt_end_index != -1:
                    # Extract text after the marker
                    start_index = prompt_end_index + len("\n<answer>\n")
                    generated_texts.append(decoded[i][start_index:].strip())
                else:
                    # Fallback if marker isn't found (shouldn't happen ideally)
                    generated_texts.append(decoded[i].strip()) 

            completions = [[{"content": txt}] for txt in generated_texts]
            solutions = [str(batch["solution"][i]) for i in range(num_examples_in_batch)] # Correct solution access

            rewards = accuracy_reward(completions, solutions)

            for r in rewards:
                if r is None:
                    continue  # skip unverifiable examples
                n_total += 1
                if r > 0.5:  # binary correctness (same heuristic as open‑r1 scripts)
                    n_correct += 1

    if n_total == 0:
        warnings.warn("No verifiable examples – returning NaN accuracy.")
        return float("nan")
    return n_correct / n_total

# ----------------------------------------------------------------------
#  Core routine
# ----------------------------------------------------------------------

def evaluate_checkpoints(exp_dir: Path, save_dir: Path, *, device: str, batch_size: int, max_new_tokens: int):
    exp_dir = exp_dir.resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── W&B setup ───────────────────────────────────────────────────────
    wandb_run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "open_r1_eval"),
        name=exp_dir.name,
        config=dict(
            exp_dir=str(exp_dir),
            save_dir=str(save_dir),
            device=str(device),
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            tasks=list(TASKS.keys()),
        ),
    )
    # -------------------------------------------------------------------

    # Discover checkpoints
    checkpoint_dirs = sorted(
        [p for p in exp_dir.glob("checkpoint-*") if p.is_dir()],
        key=lambda p: int(re.search(r"checkpoint-(\d+)", p.name).group(1)),
    )
    if not checkpoint_dirs:
        raise RuntimeError("No checkpoint-* directories found under " + str(exp_dir))

    records: List[Dict[str, float]] = []  # for DataFrame

    for ckpt in tqdm(checkpoint_dirs, desc="Checkpoints"):
        step = int(re.search(r"checkpoint-(\d+)", ckpt.name).group(1))
        print(f"\n▶ Evaluating step {step} …")

        # Instantiate dummy configs needed by get_tokenizer/get_model
        model_conf = ModelConfig(model_name_or_path=str(ckpt))
        # Minimal training config, chat_template=None is important
        train_conf = SFTConfig(output_dir="", chat_template=None) 

        tokenizer = get_tokenizer(model_conf, train_conf)
        model = get_model(model_conf, train_conf).to(device)

        row = {"step": step}
        for task, (dpath, cfg, split) in TASKS.items():
            dataset = load_dataset(dpath, cfg, split=split)
            acc = accuracy_for_dataset(model, tokenizer, dataset, device=device, batch_size=batch_size, max_new_tokens=max_new_tokens)
            row[task] = acc
            print(f"  {task:12s}: {acc:.3f}")
        records.append(row)

        # ── W&B per-checkpoint metrics ──────────────────────────────────
        wandb.log({**{f"{k}_acc": v for k, v in row.items() if k != "step"}, "step": step})
        # ----------------------------------------------------------------

        # free GPU mem between checkpoints
        del model; torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    #  Save metrics
    # ------------------------------------------------------------------
    df = pd.DataFrame(records).sort_values("step").reset_index(drop=True)
    df.to_csv(save_dir / "metrics.csv", index=False)
    (save_dir / "metrics.json").write_text(df.to_json(orient="records", indent=2))
    print("Saved metrics =>", save_dir / "metrics.csv")

    # ------------------------------------------------------------------
    #  Plots
    # ------------------------------------------------------------------
    set_neurips_style()

    # 1. Learning curves -------------------------------------------------
    fig, ax = plt.subplots(figsize=(3.5, 2.6))  # nice single‑column size
    for task in TASKS.keys():
        ax.plot(df["step"], df[task], marker="o", label=task)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Accuracy")
    ax.set_title("Checkpoint accuracy over training")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(save_dir / "learning_curve.pdf")
    # ── W&B figure log ─────────────────────────────────────────────────-
    wandb.log({"learning_curve": wandb.Image(fig)})
    # -------------------------------------------------------------------
    plt.close(fig)

    # 2. Correlation heat‑map -------------------------------------------
    corr = df[[t for t in TASKS]].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(2.6, 2.3))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Task correlations")
    fig.tight_layout()
    fig.savefig(save_dir / "task_correlation_heatmap.pdf")
    # ── W&B figure log ─────────────────────────────────────────────────-
    wandb.log({"task_correlation_heatmap": wandb.Image(fig)})
    # -------------------------------------------------------------------
    plt.close(fig)

    # 3. Scatter matrix --------------------------------------------------
    import itertools
    tasks = list(TASKS)
    n = len(tasks)
    fig, axes = plt.subplots(n, n, figsize=(2.4*n, 2.4*n))
    for i, j in itertools.product(range(n), repeat=2):
        ax = axes[i, j]
        if i == j:
            # histogram
            ax.hist(df[tasks[i]], bins=10, alpha=0.7)
        else:
            ax.scatter(df[tasks[j]], df[tasks[i]], s=15, alpha=0.8)
        if i == n-1:
            ax.set_xlabel(tasks[j])
        else:
            ax.set_xticklabels([])
        if j == 0:
            ax.set_ylabel(tasks[i])
        else:
            ax.set_yticklabels([])
    fig.suptitle("Pair‑wise checkpoint accuracies", y=0.92)
    fig.tight_layout()
    fig.savefig(save_dir / "scatter_matrix.pdf")
    # ── W&B figure log ─────────────────────────────────────────────────-
    wandb.log({"scatter_matrix": wandb.Image(fig)})
    # -------------------------------------------------------------------
    plt.close(fig)

    print("All plots saved to", save_dir)

    # ── W&B finish ─────────────────────────────────────────────────────-
    wandb_run.finish()
    # -------------------------------------------------------------------

# ----------------------------------------------------------------------
#  Entry‑point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RLHF checkpoints on multiple tasks (using open‑r1 helpers)")
    parser.add_argument("--exp-dir", type=Path, required=True, help="Experiment directory containing checkpoint-* subdirs")
    parser.add_argument("--save-dir", type=Path, required=True, help="Where to store CSV / JSON and PDF figs")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device e.g. 'cuda', 'cuda:0', 'cpu'")
    parser.add_argument("--batch-size", type=int, default=8, help="Generation batch size")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens to generate per example")
    args = parser.parse_args()

    torch_device = torch.device(args.device)
    evaluate_checkpoints(args.exp_dir, args.save_dir, device=torch_device, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)
