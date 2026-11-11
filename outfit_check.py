"""
A first check that needs no answer key: given an inspo, can the model pick out the photo of that person wearing their
version of it, from among everyone's "how I wore it" photos?

It's not the real task. The app matches single pieces against a closet, while this compares whole photos, background
and all. But it runs on the inspo/execution pairs we already have, so it's a first real number, and it shows
whether CLIP or FashionCLIP sees clothes better before anyone crops anything.

Run it:
    uv run outfit_check.py                   both models
    uv run outfit_check.py --model clip      just plain CLIP

File names decide the pairs (PLAN.md, stage 1):
    model_002_inspo_003_a.png, model_002_inspo_003_b.png    two inspos for one look
    model_002_execution_001_a.png, _b.png, _crop.png        several photos of one look
Everything with the same person and number is one look.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import torch

from baseline import MODELS, IMAGE_TYPES, SPLITS, load_model, embed

TOP_K = 3


def look_of(path):
    """model_002_inspo_003_b.png → ("model_002", "003"). The letter and _crop only say which photo of the look it is."""
    found = re.match(r"([a-z0-9_]+?)_(?:inspo|execution)_(\d{3})", path.stem)
    if not found: raise SystemExit(f"Can't tell which look {path.name} belongs to. Expected e.g. model_002_inspo_001.png")
    return found.groups()


def find_photos(data):
    inspos, executions = [], []
    for split in SPLITS:
        for kind, found in [("inspo", inspos), ("outfits", executions)]:
            for path in sorted((data / split).glob(f"*/{kind}/*")):
                if path.suffix.lower() in IMAGE_TYPES: found.append(path)
    return inspos, executions


def rank_looks(inspo_vectors, execution_vectors, execution_looks):
    """For each inspo, scores every look by its best-matching photo, then ranks the looks best first."""
    looks = sorted(set(execution_looks))
    scores = inspo_vectors @ execution_vectors.T  # Cosine similarity, since every vector has length 1.
    per_look = np.stack([scores[:, [i for i, look in enumerate(execution_looks) if look == target]].max(axis=1) for target in looks], axis=1)
    return looks, per_look


def bootstrap(ranks, groups, rounds=1000):
    """95% ranges, resampling whole looks rather than single inspos, because two inspos for one look aren't two
    independent tries."""
    random = np.random.default_rng(0)
    unique = sorted(set(groups))
    by_group = {g: [r for r, gr in zip(ranks, groups) if gr == g] for g in unique}
    samples = []
    for _ in range(rounds):
        picked = np.concatenate([by_group[unique[i]] for i in random.integers(0, len(unique), len(unique))])
        samples.append([np.mean(picked == 1), np.mean(picked <= TOP_K), np.mean(1 / picked)])
    return np.percentile(samples, [2.5, 97.5], axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[*MODELS, "both"], default="both")
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent.parent / "Style.io_Data")
    args = parser.parse_args()

    inspos, executions = find_photos(args.data)
    execution_looks = [look_of(path) for path in executions]
    missing = {look_of(path) for path in inspos} - set(execution_looks)
    if missing: raise SystemExit(f"These inspos have no execution photo yet: {sorted(missing)}")

    looks_total = len(set(execution_looks))
    chance = [1 / looks_total, min(TOP_K, looks_total) / looks_total, np.mean(1 / np.arange(1, looks_total + 1))]
    print(f"{len(inspos)} inspos, {len(executions)} execution photos, {looks_total} looks")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    for name in (MODELS if args.model == "both" else [args.model]):
        model, processor = load_model(name, device)
        looks, per_look = rank_looks(embed(inspos, model, processor, device), embed(executions, model, processor, device), execution_looks)

        print(f"\n=== {name} ===")
        ranks, groups = [], []
        for path, row in zip(inspos, per_look):
            order = np.argsort(-row)
            ranked = [looks[i] for i in order]
            rank = ranked.index(look_of(path)) + 1
            ranks.append(rank)
            groups.append(look_of(path))
            guesses = "  ".join(f"{p}_{n} {row[i]:.3f}" for (p, n), i in zip(ranked[:TOP_K], order[:TOP_K]))
            print(f"  {'✓' if rank == 1 else ' '} {path.stem:<18} right look ranked {rank}   top 3: {guesses}")

        ranks = np.array(ranks)
        score = [np.mean(ranks == 1), np.mean(ranks <= TOP_K), np.mean(1 / ranks)]
        low, high = bootstrap(ranks, groups)
        print()
        for i, label in enumerate(["Recall@1", f"Recall@{TOP_K}", "MRR"]):
            print(f"  {label:<9} {score[i]:.2f} ({low[i]:.2f}–{high[i]:.2f})   random: {chance[i]:.2f}")


if __name__ == "__main__":
    main()
