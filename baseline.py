"""
Stage 2 baseline: how well does a pretrained model match Pinterest pieces to closet items, with no training at all?

Run it:
    uv run baseline.py                      both models
    uv run baseline.py --model clip         just plain CLIP
    uv run baseline.py --data some/folder   a different data folder
    uv run baseline.py --crops redrawn      the product-shot redraws instead of the rough crops
    uv run baseline.py --crops clean        the cut-outs from cutout.py
    uv run baseline.py --closet own         match each piece against its owner's closet only

Data layout (PLAN.md, stage 1). The data lives outside this repo, in the Style.io_Data folder next to it, so
people's photos can't end up on GitHub. One folder per person, because each person's pieces are only matched
against their own closet:

    Style.io_Data/tune/<friend>/closet/<category>/<item>.jpg     friends: for choosing settings like the cutoff
    Style.io_Data/tune/<friend>/pieces/<piece>.jpg
    Style.io_Data/tune/<friend>/answers.csv                      piece,category,correct_item
    Style.io_Data/test/josh/...                                  Josh: the final grade, never used for choosing settings

Those are the rough rectangle crops, scored by default. Other versions of the same crops, with the same names, sit in
<person>/redrawn/ (Codex's product-shot redraws) and <person>/clean/ (cutout.py's cut-outs), each with its own
pieces/ and closet/ folders.

Groups come from Style.io_Data/split.json (train, validation, test, by person). Anyone not listed counts as train.

Closets: most Instagram people only own what they wore in their reel, often one item per category, which makes
matching against their own closet trivial. So by default ("plus-train") every piece is matched against the closets of
its whole group plus the train group's, about 100 items for validation and test. "group" pools only its own group;
"own" keeps the old per-person closets.

It prints Recall@1, Recall@3 and MRR with 95% ranges for each group, next to what random guessing would score.
Every piece's result also goes to results/<model>-<crops>-<closet>-pieces.csv.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import CLIPModel, CLIPProcessor

MODELS = {
    "clip": "openai/clip-vit-base-patch32",
    "fashion-clip": "patrickjohncyh/fashion-clip",
}
CATEGORIES = ["top", "bottoms", "dress", "shoes", "layer", "headwear"]
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp"}
SPLITS = ["train", "validation", "test"]
TOP_K = 3  # The app shows the top 3, so that's the number that matters.


# ---------- Reading the data ----------

def find_people(data):
    """Every person folder, tagged with its group from split.json. Anyone not listed there trains."""
    group_of = {}
    if (data / "split.json").exists():
        split = json.loads((data / "split.json").read_text())
        group_of = {person: group for group in SPLITS for person in split.get(group, [])}
    people = []
    for folder in ["tune", "test"]:
        if not (data / folder).exists(): continue
        for person in sorted((data / folder).iterdir()):
            if person.is_dir(): people.append((group_of.get(person.name, "train"), person))
    return people


def load_closet(person):
    """Every closet photo, with its category taken from the folder it's in: closet/top/black-tee.jpg."""
    closet = []
    for category in CATEGORIES:
        folder = person / "closet" / category
        if not folder.exists(): continue
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() in IMAGE_TYPES:
                closet.append({"name": path.stem, "category": category, "path": path})

    names = [item["name"] for item in closet]
    repeats = {name for name in names if names.count(name) > 1}
    if repeats: raise SystemExit(f"{person}: two closet photos share a name, so answers.csv can't tell them apart: {repeats}")
    return closet


def load_answers(person, crops, closet):
    """The answer key. Checks every row up front, so a typo stops the run instead of quietly lowering the score."""
    # Someone still collecting photos has no answer key yet, so they're skipped for now instead of stopping the run.
    if not (person / "answers.csv").exists(): return []
    category_of = {item["name"]: item["category"] for item in closet}
    rows = []
    with open(person / "answers.csv", newline="") as file:
        for line, row in enumerate(csv.DictReader(file), start=2):
            row = {key: value.strip() for key, value in row.items()}
            where = f"{person.name}/answers.csv line {line}"
            if not (crops / "pieces" / row["piece"]).exists(): raise SystemExit(f"{where}: no file {crops / 'pieces' / row['piece']}")
            if row["category"] not in CATEGORIES: raise SystemExit(f"{where}: category must be one of {CATEGORIES}")
            if row["correct_item"] != "none":
                if row["correct_item"] not in category_of: raise SystemExit(f"{where}: no closet photo named {row['correct_item']}")
                if category_of[row["correct_item"]] != row["category"]:
                    raise SystemExit(f"{where}: {row['correct_item']} is in {category_of[row['correct_item']]}, not {row['category']}")
            rows.append(row)
    return rows


# ---------- Turning images into numbers ----------

def load_model(name, device):
    model = CLIPModel.from_pretrained(MODELS[name]).to(device).eval()
    return model, CLIPProcessor.from_pretrained(MODELS[name])


def load_image(path):
    # Phone photos are often stored sideways with a note saying which way is up. Without this, the model sees them sideways.
    image = ImageOps.exif_transpose(Image.open(path))
    # Cut-outs are transparent around the garment. A plain convert to RGB throws the transparency away and the
    # original background comes back, so they go onto white first. Pieces and closet items get the same white.
    if image.mode in ("RGBA", "LA", "P"):
        image = Image.alpha_composite(Image.new("RGBA", image.size, "white"), image.convert("RGBA"))
    return image.convert("RGB")


def embed(paths, model, processor, device):
    """Closet photos and Pinterest crops both come through here, so they're always processed the same way (rule 1)."""
    return embed_images([load_image(path) for path in paths], model, processor, device)


def embed_images(images, model, processor, device):
    """The same as embed, for images already open (train.py uses it for flipped and re-cropped copies)."""
    vectors = []
    for start in range(0, len(images), 16):
        batch = images[start:start + 16]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.get_image_features(**inputs)
        if not torch.is_tensor(output): output = output.pooler_output  # Newer transformers wraps the numbers in an object.
        vectors.append(output.float().cpu().numpy())
    vectors = np.concatenate(vectors)
    # Making every vector length 1, so a plain dot product is the cosine similarity (rule 6).
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


# ---------- Matching and scoring ----------

def pool_people(group_of, group, mode, person):
    """Whose closet items a piece is ranked against.
    own: just its owner's. group: everyone in its group. plus-train: its group plus the train group, so validation
    and test pieces face a closet of about 100 items, closer to a real one."""
    if mode == "own": return {person}
    keep = {group, "train"} if mode == "plus-train" else {group}
    return {name for name, g in group_of.items() if g in keep}


def match(answers, piece_vectors, closet, closet_vectors):
    """For each piece, ranks the closet items in the same category, best first."""
    results = []
    for row, piece_vector in zip(answers, piece_vectors):
        candidates = [i for i, item in enumerate(closet) if item["category"] == row["category"]]
        if not candidates:
            # A "no match" piece in a category nobody here owns: nothing to rank, and nothing wrong either.
            if row["correct_item"] == "none":
                results.append({**row, "top_guess": None, "top_score": None, "rank": None, "candidates": 0})
                continue
            raise SystemExit(f"{row['piece']}: no {row['category']} in the closet to match against")
        scores = closet_vectors[candidates] @ piece_vector
        order = np.argsort(-scores)
        # Items are told apart by owner and name, since pooled closets mix several people.
        ranked = [(closet[candidates[i]]["person"], closet[candidates[i]]["name"]) for i in order]
        answer = (row["person"], row["correct_item"])
        results.append({
            **row,
            "top_guess": "/".join(ranked[0]),
            "top_score": round(float(scores[order[0]]), 4),
            "rank": ranked.index(answer) + 1 if row["correct_item"] != "none" else None,
            "candidates": len(candidates),
        })
    return results


def metrics(results):
    """Recall@1, Recall@3 and MRR, worked out per category then averaged, so shoes count as much as tops."""
    by_category = {}
    for result in results:
        by_category.setdefault(result["category"], []).append(result["rank"])
    per_category = []
    for ranks in by_category.values():
        ranks = np.array(ranks)
        per_category.append([np.mean(ranks == 1), np.mean(ranks <= TOP_K), np.mean(1 / ranks)])
    return np.mean(per_category, axis=0)


def random_guessing(results):
    """What you'd score picking closet items at random. With 5 shoes, a random pick is right 1 time in 5."""
    by_category = {}
    for result in results:
        by_category.setdefault(result["category"], []).append(result["candidates"])
    per_category = []
    for sizes in by_category.values():
        sizes = np.array(sizes)
        average_reciprocal_rank = np.array([np.mean(1 / np.arange(1, n + 1)) for n in sizes])
        per_category.append([np.mean(1 / sizes), np.mean(np.minimum(TOP_K, sizes) / sizes), np.mean(average_reciprocal_rank)])
    return np.mean(per_category, axis=0)


def ranges(results, rounds=1000):
    """95% ranges by bootstrapping: re-score random resamples of the pieces and keep the middle 95%."""
    random = np.random.default_rng(0)  # Same seed every run, so the ranges don't wobble between runs.
    samples = [metrics([results[i] for i in random.integers(0, len(results), len(results))]) for _ in range(rounds)]
    return np.percentile(samples, [2.5, 97.5], axis=0)


# ---------- Report ----------

def report(name, results):
    print(f"\n=== {name} ===")
    for split in SPLITS:
        matched = [r for r in results if r["split"] == split and r["rank"] is not None]
        if not matched:
            print(f"  {split}: no pieces with a correct item")
            continue
        score, (low, high), chance = metrics(matched), ranges(matched), random_guessing(matched)
        people = sorted({r["person"] for r in matched})
        print(f"  {split}: {len(matched)} pieces from {', '.join(people)}")
        for i, label in enumerate(["Recall@1", f"Recall@{TOP_K}", "MRR"]):
            print(f"    {label:<9} {score[i]:.2f} ({low[i]:.2f}–{high[i]:.2f})   random: {chance[i]:.2f}")

    small = sorted({f"{r['person']}/{r['category']}" for r in results if r["candidates"] <= TOP_K})
    if small: print(f"  Heads up: {small} have {TOP_K} or fewer closet items, so Recall@{TOP_K} is always 1 there.")

    right = [r["top_score"] for r in results if r["rank"] == 1]
    none = [r["top_score"] for r in results if r["correct_item"] == "none" and r["top_score"] is not None]
    if right and none:
        print(f"  Top score when the first guess was right: {np.mean(right):.3f}. On 'no match' pieces: {np.mean(none):.3f}.")
        print("  The bigger that gap, the easier the cutoff in stage 5.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[*MODELS, "both"], default="both")
    # The data folder sits next to this repo, never inside it.
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent.parent / "Style.io_Data")
    parser.add_argument("--crops", choices=["rough", "redrawn", "clean"], default="rough")
    parser.add_argument("--closet", choices=["plus-train", "group", "own"], default="plus-train")
    args = parser.parse_args()

    # Reading and checking everything before loading a model, so a typo shows up in a second, not after a download.
    people = []
    for split, person in find_people(args.data):
        crops = person if args.crops == "rough" else person / args.crops
        closet = load_closet(crops)
        answers = load_answers(person, crops, closet)
        for item in closet: item["person"] = person.name
        for row in answers: row["person"] = person.name
        if answers: people.append((split, person, crops, closet, answers))
    if not people: raise SystemExit(f"No answers yet in {args.data}/tune/<person>/ or {args.data}/test/<person>/")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)

    for name in (MODELS if args.model == "both" else [args.model]):
        model, processor = load_model(name, device)
        results = []
        # Every closet item is embedded once, then pooled per group or kept per person.
        everything = [item for _, _, _, closet, _ in people for item in closet]
        vectors = embed([item["path"] for item in everything], model, processor, device)
        for split, person, crops, closet, answers in people:
            group_of = {p.name: s for s, p, *_ in people}
            names = pool_people(group_of, split, args.closet, person.name)
            pool = [i for i, item in enumerate(everything) if item["person"] in names]
            piece_vectors = embed([crops / "pieces" / row["piece"] for row in answers], model, processor, device)
            for result in match(answers, piece_vectors, [everything[i] for i in pool], vectors[pool]):
                results.append({"split": split, **result})
        report(f"{name}, {args.crops} crops, {args.closet} closets", results)

        with open(out / f"{name}-{args.crops}-{args.closet}-pieces.csv", "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(dict.fromkeys(key for r in results for key in r)))
            writer.writeheader()
            writer.writerows(results)


if __name__ == "__main__":
    main()
