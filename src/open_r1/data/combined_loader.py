"""open-r1 extra dataset loader"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import tempfile
import textwrap
import atexit
from typing import List, Dict, Any

import datasets
from datasets import DatasetDict, Dataset, Features, Value

# Optional – these imports are only needed for a few datasets; we import lazily
try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _ensure(msg: str, cond: bool):
    if not cond:
        raise RuntimeError(msg)


def _map(ds: Dataset, fn, remove_columns: List[str] | None = None):
    """Convenience wrapper that preserves formatting + avoids copying columns."""
    keep = ds.column_names if remove_columns is None else remove_columns
    return ds.map(fn, remove_columns=keep, num_proc=os.cpu_count() or 1)


# ---------------------------------------------------------------------------
# Individual dataset loaders – each returns **DatasetDict** with the required
# columns in place.  When possible we stream from the Hub to minimise disk use.
# ---------------------------------------------------------------------------


# 1. Advent of Code ----------------------------------------------------------

def _load_advent_of_code() -> DatasetDict:
    ds = datasets.load_dataset("isavita/advent-of-code", split="train", streaming=False)

    def fmt(ex):
        prompt = f"{ex['description'].strip()}\n\nInput:\n{ex['input']}"
        return {
            "prompt": prompt,
            "solution": ex["answer"].strip() if isinstance(ex["answer"], str) else str(ex["answer"]),
        }

    ds = _map(ds, fmt)
    return DatasetDict({"train": ds})


# 2. Bespoke‑Stratos ---------------------------------------------------------

def _load_bespoke_stratos() -> DatasetDict:
    ds = datasets.load_dataset("bespokelabs/Bespoke-Stratos-17k", split="train", streaming=False)

    def fmt(ex):
        # Solution text already ends with the final answer after "Answer:" or is boxed.
        match = re.search(r"Answer\s*[:：]\s*(.+)$", ex["solution"], flags=re.M | re.S)
        final = match.group(1).strip() if match else ex["solution"].split("\n")[-1].strip()
        prompt = textwrap.dedent(ex["question"]).strip()
        return {"prompt": prompt, "solution": final}

    return DatasetDict({"train": _map(ds, fmt)})


# 3. BIG‑Bench (requires `config`) -----------------------------------------

def _load_bigbench(config_name: str) -> DatasetDict:
    _ensure("A BIG‑bench task (config) name is required.", bool(config_name))
    # Use the lighter tasksource fork to avoid heavy deps.
    ds = datasets.load_dataset("tasksource/bigbench", config_name, split="train", streaming=False)

    if "answer" in ds.column_names:
        def fmt(ex):
            return {"prompt": ex["inputs"], "solution": ex["answer"]}
    else:  # multiple‑choice style tasks
        def fmt(ex):
            correct = ex["target_scores"].index(max(ex["target_scores"]))
            return {"prompt": ex["inputs"], "solution": ex["targets"][correct]}

    return DatasetDict({"train": _map(ds, fmt)})


# 4. BrainTeaser ------------------------------------------------------------

def _load_brainteaser() -> DatasetDict:
    ds = datasets.load_dataset("ErfanMoosaviMonazzah/brain-teasers", split="train", streaming=False)

    def fmt(ex):
        prompt = ex["question"] + "\nChoices: " + " | ".join(ex["choices"])
        return {"prompt": prompt, "solution": ex["answer"]}

    return DatasetDict({"train": _map(ds, fmt)})


# 5. NYT Connections --------------------------------------------------------

def _load_connections() -> DatasetDict:
    ds = datasets.load_dataset("eric27n/NYT-Connections", split="train", streaming=False)

    # group by puzzle_id to collapse 4 category rows into one example
    grouped: Dict[str, Dict[str, Any]] = {}
    for row in ds:
        pid = row["puzzle_id"]
        grouped.setdefault(pid, {"words": [], "solution_groups": []})
        grouped[pid]["solution_groups"].append({"name": row["label"], "members": row["text"].split()})
        grouped[pid]["words"].extend(row["text"].split())

    def gen_items():
        for i, (pid, info) in enumerate(grouped.items()):
            prompt = "Solve the NYT Connections puzzle. Group the 16 words into 4 categories."\
                     f"\nWords: {' '.join(info['words'])}"
            yield i, {"prompt": prompt, "solution": json.dumps(info["solution_groups"])}

    features = Features({"prompt": Value("string"), "solution": Value("string")})
    ds_out = Dataset.from_generator(gen_items, features=features)
    return DatasetDict({"train": ds_out})


# 6. Countdown (generate + solve) ------------------------------------------

def _countdown_solve(nums: List[int], target: int) -> str | None:
    """Brute‑force search up to 4‑number problems; returns a valid expression or None."""

    import itertools
    import operator

    ops = [(operator.add, "+"), (operator.sub, "-"), (operator.mul, "*"), (operator.floordiv, "/")]

    def eval_pair(a, b):
        for op, sym in ops:
            if sym == "/" and b == 0:
                continue
            yield op(a, b), f"({a}{sym}{b})"

    numbers = nums
    expressions = {n: str(n) for n in numbers}
    best = None
    best_expr = None

    for perm in itertools.permutations(numbers):
        a, b, c, d, *_ = perm + (None,) * (4 - len(perm))
        stack = [(a, str(a))]
        for n in (b, c, d):
            if n is None:
                break
            new_stack = []
            for val, expr in stack:
                for res, subexpr in eval_pair(val, n):
                    new_stack.append((res, subexpr.replace(str(val), expr)))
            stack = new_stack
        for val, expr in stack:
            if val == target:
                return expr
            if best is None or abs(val - target) < abs(best - target):
                best, best_expr = val, expr
    return best_expr if best == target else None


def _load_countdown() -> DatasetDict:
    base = datasets.load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train", streaming=False)

    rows = []
    for ex in base:
        nums = ex["nums"]
        target = ex["target"]
        sol = _countdown_solve(nums, target)
        if sol is None:
            # skip unsolved – open‑r1 rewards collapse without an answer
            continue
        rows.append({
            "prompt": f"Numbers: {nums} Target: {target}. Give any valid expression that equals the target.",
            "solution": sol,
            "verification_info": {
                "language": "python",
                "test_cases": [{"input": "", "output": str(target)}],
            },
        })

    ds_out = Dataset.from_list(rows)
    return DatasetDict({"train": ds_out})


# 7. GSM8K ------------------------------------------------------------------

def _load_gsm8k(split="train") -> DatasetDict:
    ds = datasets.load_dataset("openai/gsm8k", "main", split=split, streaming=False)

    def fmt(ex):
        # extract the final `#### answer` token
        match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", ex["answer"])
        return {"prompt": ex["question"], "solution": match.group(1) if match else ex["answer"]}

    return DatasetDict({split: _map(ds, fmt)})


# 8. HumanEval --------------------------------------------------------------

def _load_humaneval() -> DatasetDict:
    ds = datasets.load_dataset("openai/openai_humaneval", split="test", streaming=False)

    def fmt(ex):
        prompt = ex["prompt"]
        solution_code = ex["canonical_solution"]
        verification = {
            "language": "python",
            "test_cases": [{"input": "", "output": ""}],  # tests are included downstream via HiddenIO
        }
        return {"prompt": prompt, "solution": solution_code, "verification_info": verification}

    return DatasetDict({"test": _map(ds, fmt)})


# 9. LogiQA -----------------------------------------------------------------

def _load_logiqa() -> DatasetDict:
    ds = datasets.load_dataset("lucasmccabe/logiqa", split="train", streaming=False)

    def fmt(ex):
        choices = ex["options"]
        answer_text = choices[ex["label"]]
        prompt = ex["context"] + "\n" + ex["question"] + "\nChoices: " + " | ".join(choices)
        return {"prompt": prompt, "solution": answer_text}

    return DatasetDict({"train": _map(ds, fmt)})


# 10. MacGyver --------------------------------------------------------------

def _load_macgyver() -> DatasetDict:
    _ensure("pandas is required for MacGyver", pd is not None)
    url = "https://raw.githubusercontent.com/allenai/MacGyver/main/data/problem_solution_pair.csv"
    df = pd.read_csv(url)
    ds = datasets.Dataset.from_pandas(df[["Problem", "Solution"]].rename(columns={"Problem": "prompt", "Solution": "solution"}))
    return DatasetDict({"train": ds})


# 11. MATH ------------------------------------------------------------------

def _load_math(split="train") -> DatasetDict:
    ds = datasets.load_dataset("hendrycks/competition_math", split=split, streaming=False)

    def fmt(ex):
        # Keep full solution; final answer can be parsed with regex if needed
        return {"prompt": ex["problem"], "solution": ex["solution"]}

    return DatasetDict({split: _map(ds, fmt)})


# 12. Multiply (synthetic) --------------------------------------------------

def _load_multiply(size: int = 10000, n_digits: int = 3) -> DatasetDict:
    rows = []
    for _ in range(size):
        a = random.randint(10 ** (n_digits - 1), 10 ** n_digits - 1)
        b = random.randint(10 ** (n_digits - 1), 10 ** n_digits - 1)
        prompt = f"What is {a} * {b}?"
        rows.append({"prompt": prompt, "solution": str(a * b)})
    ds = Dataset.from_list(rows)
    return DatasetDict({"train": ds})


# 13. Numina Math TIR -------------------------------------------------------

def _load_numina_math() -> DatasetDict:
    ds = datasets.load_dataset("AI-MO/NuminaMath-TIR", split="train", streaming=False)

    def fmt(ex):
        return {"prompt": ex["problem"], "solution": ex["solution"]}

    return DatasetDict({"train": _map(ds, fmt)})


# 14. Project Euler ---------------------------------------------------------

def _load_project_euler() -> DatasetDict:
    _ensure("pandas is required for Project Euler", pd is not None)
    kaggle_csv = "https://huggingface.co/datasets/OpenR1-cdn/peuler/resolve/main/problems_answers.csv"
    df = pd.read_csv(kaggle_csv)
    df = df.rename(columns={"question": "prompt", "answer": "solution"})
    ds = datasets.Dataset.from_pandas(df[["prompt", "solution"]])
    return DatasetDict({"train": ds})


# 15. RiddleSense -----------------------------------------------------------

def _load_riddlesense(split="train") -> DatasetDict:
    ds = datasets.load_dataset("INK-USC/riddle_sense", split=split, streaming=False)

    def fmt(ex):
        prompt = ex["question"] + "\nChoices: " + " | ".join([c["text"] for c in ex["choices"]])
        correct = next(c["text"] for c in ex["choices"] if c["label"] == ex["answerKey"])
        return {"prompt": prompt, "solution": correct}

    return DatasetDict({split: _map(ds, fmt)})


# 16. SPLAT -----------------------------------------------------------------

def _load_splat() -> DatasetDict:
    _ensure("pandas is required for SPLAT", pd is not None)
    url = "https://raw.githubusercontent.com/lysandrejik/splat/main/data/puzzles.xlsx"
    df = pd.read_excel(url)
    df = df.rename(columns={"Puzzle": "prompt", "Solution": "solution"})
    ds = datasets.Dataset.from_pandas(df[["prompt", "solution"]])
    return DatasetDict({"train": ds})


# ---------------------------------------------------------------------------
# Registry mapping + dispatch
# ---------------------------------------------------------------------------

_DATASET_LOADERS = {
    "advent_of_code": _load_advent_of_code,
    "bespoke_stratos": _load_bespoke_stratos,
    "bigbench": _load_bigbench,
    "brainteaser": _load_brainteaser,
    "connections": _load_connections,
    "countdown": _load_countdown,
    "gsm8k": _load_gsm8k,
    "humaneval": _load_humaneval,
    "logiqa": _load_logiqa,
    "macgyver": _load_macgyver,
    "math": _load_math,
    "multiply": _load_multiply,
    "numina_math_tir": _load_numina_math,
    "project_euler": _load_project_euler,
    "riddlesense": _load_riddlesense,
    "splat": _load_splat,
}


# ---------------------------------------------------------------------------
# 🤗 Datasets integration – a *single* builder that proxies to the helpers above
# ---------------------------------------------------------------------------

class CombinedLoader(datasets.GeneratorBasedBuilder):
    """A meta‑dataset builder that proxies to the individual loaders."""

    BUILDER_CONFIG_CLASS = datasets.BuilderConfig

    BUILDER_CONFIGS = [datasets.BuilderConfig(name=name, version=datasets.Version("1.0.0")) for name in _DATASET_LOADERS]

    DEFAULT_CONFIG_NAME = "advent_of_code"

    def _info(self):
        return datasets.DatasetInfo(
            description="A union wrapper exposing many math/reasoning/code datasets in open‑r1 compatible format.",
            features=datasets.Features(
                {
                    "prompt": Value("string"),
                    # solution may be long text or code
                    "solution": Value("string"),
                    # optional – present only for code tasks
                    "verification_info": datasets.features.Value("string"),
                }
            ),
            homepage="https://github.com/huggingface/open-r1",
        )

    def _split_generators(self, dl_manager):  # noqa: D401
        name = self.config.name
        if name not in _DATASET_LOADERS:
            raise ValueError(f"Unknown dataset key: {name}. Available: {list(_DATASET_LOADERS)}")

        # Call the underlying loader. We *don't* stream because most helpers load
        # into memory already; if you want streaming just rewrite the helper.
        ds_dict: DatasetDict = _DATASET_LOADERS[name]()  # type: ignore

        # Honour the splits that the helper returns.
        splits = []
        for split_name, split_ds in ds_dict.items():
            # datasets ≥3.x: mktemp_dir was renamed to a private helper
            try:
                path = dl_manager._mktemp_dir(f"{name}-{split_name}")  # already exists
            except AttributeError:
                # Fallback for any future refactor
                path = tempfile.mkdtemp(prefix=f"{name}-{split_name}-")
                # optional: register a cleanup to keep the machine tidy
                atexit.register(shutil.rmtree, path, ignore_errors=True)

            split_ds.to_json(os.path.join(path, "data.jsonl"), orient="records", lines=True)
            splits.append(
                datasets.SplitGenerator(name=getattr(datasets.Split, split_name.upper(), split_name), gen_kwargs={"file": os.path.join(path, "data.jsonl")})
            )
        return splits

    def _generate_examples(self, file):  # noqa: D401
        with open(file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                obj = json.loads(line)
                yield idx, obj


# Entrypoint for `datasets.load_dataset("combined_loader.py", name)`

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=_DATASET_LOADERS.keys())
    args = parser.parse_args()

    dsdict = _DATASET_LOADERS[args.dataset]()
    print(dsdict)
