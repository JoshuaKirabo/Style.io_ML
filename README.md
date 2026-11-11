# Style.io ML

The matching side of [Style.io](https://github.com/JoshuaKirabo/Style.io_Client): drop in an outfit from Pinterest,
and for each piece in it (top, bottoms, dress, shoes, layer, headwear) get the closest pieces you already own, or "no match".

## How matching works

```
outfit photo → find each piece → crop the real pixels → turn each crop into numbers (an embedding)
             → compare with every closet piece in the same category → top 3, or "no match" below a cutoff
```

Closet photos are turned into numbers once, when they're added. Adding clothes never means retraining, so the same
model works on closets it has never seen.

## What's here

| File | What it does |
| --- | --- |
| `baseline.py` | Scores pretrained CLIP and FashionCLIP on a hand-labelled test set, with no training |
| `outfit_check.py` | A first check with no answer key: can the model pick out each inspo's "how I wore it" photo from everyone's? Whole photos, no crops |

The baseline is the number every later change has to beat. Each score is reported with a 95% range (bootstrapped)
and next to what random guessing would get, per category and averaged across categories, so a small test set can't
oversell itself.

## Run it

```bash
uv sync               # Python 3.12, PyTorch, Transformers
uv run baseline.py    # both models; --model clip or --model fashion-clip for one
```

On a Mac it uses the GPU (MPS) when it's there.

## Data

Not in this repo: it's people's photos. By default `baseline.py` reads a `Style.io_Data` folder sitting next to this
repo (or pass `--data <folder>`). The layout it expects, one folder per person:

```
Style.io_Data/tune/<person>/   used to choose settings like the "no match" cutoff
Style.io_Data/test/<person>/   the final grade, never used to choose settings
  closet/<category>/  the person's clothes: closet/shoes/white-af1.jpg
  pieces/             pieces cropped from outfit inspo photos
  answers.csv         piece,category,correct_item   ("none" when they own nothing like it)
```

Each person's pieces are only matched against their own closet.

## Results

None yet. Scores go here once the test set is built.
