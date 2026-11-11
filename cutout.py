"""
Turns rough rectangle crops into clean cut-outs: the garment's real pixels with the background removed. Nothing is
redrawn or recoloured. It runs on this Mac, so no photo leaves it.

Run it after Codex has made the crops and each person's boxes.json:
    uv run cutout.py
    uv run cutout.py --data some/folder

How: SAM (Meta's Segment Anything) gets the original photo and the garment's box, and returns which pixels inside
the box belong to the object the box frames. Those pixels are kept and everything else becomes transparent.

Reads <person>/boxes.json, a list of
    {"crop": "pieces/model_001_inspo_003_top.png", "source": "inspo/model_001_inspo_003.png", "box": [left, top, right, bottom]}
and writes the cut-out to <person>/clean/<crop>, with the same name, so answers.csv works for both sets.
Also writes <person>/clean/review.html: each rough crop next to its cut-out, to check by eye.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor

from baseline import find_people, load_image

SAM = "facebook/sam-vit-base"
MARGIN = 8  # Pixels kept around the garment, so edges aren't cut flush.


def cut_out(image, box, model, processor, device):
    """The garment inside the box, as an RGBA image cropped to the garment, plus how sure SAM was."""
    inputs = processor(image, input_boxes=[[box]], return_tensors="pt")
    # The two sizes are for scaling SAM's answer back up to the photo's size, not for the model itself.
    sizes = inputs.pop("original_sizes"), inputs.pop("reshaped_input_sizes")
    # The box comes out as 64-bit numbers, which the Mac's GPU can't use, so everything goes to 32-bit.
    inputs = {key: value.to(device, torch.float32) for key, value in inputs.items()}
    with torch.no_grad():
        output = model(**inputs, multimask_output=True)
    # SAM gives three guesses at different sizes (a sleeve, the top, the whole person). Keeping the one it rates best.
    masks = processor.image_processor.post_process_masks(output.pred_masks.cpu(), *sizes)[0][0]
    scores = output.iou_scores[0, 0].float().cpu().numpy()
    best = int(np.argmax(scores))
    mask = masks[best].numpy()

    # Anything outside the box isn't this garment, even if SAM spilled over into it.
    left, top, right, bottom = box
    inside = np.zeros_like(mask)
    inside[top:bottom, left:right] = True
    mask = mask & inside
    if not mask.any(): return None, float(scores[best])

    rows, columns = np.where(mask)
    crop = (
        max(columns.min() - MARGIN, 0),
        max(rows.min() - MARGIN, 0),
        min(columns.max() + MARGIN + 1, image.width),
        min(rows.max() + MARGIN + 1, image.height),
    )
    cut = image.convert("RGBA")
    cut.putalpha(Image.fromarray((mask * 255).astype(np.uint8)))
    return cut.crop(crop), float(scores[best])


def review_page(person, done):
    """One row per crop: Codex's rough rectangle, then the cut-out on a checkerboard so missing bits stand out."""
    rows = []
    for crop, score in done:
        rough = f"../{crop}"
        clean = crop
        flag = " low" if score < 0.85 else ""
        rows.append(f'<div class="row{flag}"><img src="{rough}"><img class="cut" src="{clean}"><p>{crop}<br>SAM {score:.2f}</p></div>')
    page = f"""<!doctype html><meta charset="utf-8"><title>{person.name} cut-outs</title>
<style>
  body {{ font: 14px system-ui; margin: 16px; background: #fff; color: #111; }}
  .row {{ display: flex; gap: 16px; align-items: center; padding: 8px 0; border-bottom: 1px solid #ddd; }}
  .row.low {{ background: #fff4e5; }}
  img {{ height: 220px; width: auto; }}
  .cut {{ background: repeating-conic-gradient(#ccc 0 25%, #fff 0 50%) 0 0 / 16px 16px; }}
</style>
<h1>{person.name}: rough crop, then cut-out</h1>
<p>Orange rows are ones SAM was less sure of. Check those first.</p>
{''.join(rows)}"""
    (person / "clean" / "review.html").write_text(page)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent.parent / "Style.io_Data")
    args = parser.parse_args()

    people = [person for _, person in find_people(args.data) if (person / "boxes.json").exists()]
    if not people: raise SystemExit(f"No boxes.json yet in {args.data}/tune/<person>/ or {args.data}/test/<person>/")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SamModel.from_pretrained(SAM).to(device).eval()
    processor = SamProcessor.from_pretrained(SAM)

    for person in people:
        entries = json.loads((person / "boxes.json").read_text())
        done, failed = [], []
        image, source = None, None
        for entry in entries:
            # Several crops come from one photo, so each photo is only opened once.
            if entry["source"] != source:
                source = entry["source"]
                image = load_image(person / source)
            box = [int(round(v)) for v in entry["box"]]
            cut, score = cut_out(image, box, model, processor, device)
            if cut is None:
                failed.append(entry["crop"])
                continue
            out = person / "clean" / entry["crop"]
            out.parent.mkdir(parents=True, exist_ok=True)
            cut.save(out)
            done.append((entry["crop"], score))

        review_page(person, done)
        low = sum(score < 0.85 for _, score in done)
        print(f"{person.name}: {len(done)} cut-outs, {low} to check closely, {len(failed)} failed {failed if failed else ''}")
        print(f"  Review: {person / 'clean' / 'review.html'}")


if __name__ == "__main__":
    main()
