# FYP-2 · Segmentation-only step (documentation)

File: `fyp2_segmentation_only.py`, written as Colab cells (`# %%`). Not run locally. Paste each cell into Colab, or upload it and run it.

## What it does
Turns 300 recordings (1 sample per woman) into thousands of **real-signal windows**. It uses no VMD, DWT or Butterworth filtering, and no synthetic data.

| Cell | Step |
|---|---|
| 1 | Install `wfdb`, mount Drive, set parameters |
| 2 | Load raw electrodes S1, S2, S3 (channel names `1`, `2`, `3`; the 9 other columns are pre-filtered copies and are ignored). The label is `gestation < 37`, and gestation is **never** an input |
| 3 | Assign every **woman** to 1 of 5 stratified folds *before* any windowing |
| 4 | Cut windows: 120 s long, first and last 180 s trimmed. **Eval windows**: stride 60 s for everyone. **Train windows**: stride 60 s for term, **15 s for preterm** (class-aware overlap, so preterm gets about 4× more windows) |
| 5 | `get_fold(k)` returns train / val / test arrays. It asserts that no woman appears in two splits |
| 6 | `RandomCropSampler`: balanced (50% preterm) random 120 s crops from train women only, for on-the-fly expansion |
| 7-8 | Sanity print for fold 0, and save `meta.csv` / `windows.csv` |

## Parameters (cell 1)
`WIN_S=120`, `TRIM_S=180`, `EVAL_STRIDE_S=60`, `TRAIN_STRIDE_S={0:60, 1:15}`, `N_FOLDS=5`. Make the preterm stride smaller for more preterm windows.

## Leakage rules built in
1. Folds are assigned per woman first, so a woman's windows never span train and test.
2. Class-aware stride and random crops apply to train women only. Val and test use the plain uniform stride.
3. Normalisation is per window, so no statistics are shared between women.
4. Evaluate **per woman**: `aggregate_by_woman(window_probs, groups)` before computing AUC or sensitivity.

## Usage
```python
d = get_fold(0)                          # repeat for 0..4
Xtr, ytr, gtr = d['train']               # shape (n_windows, 3, 2400)
Xva, yva, gva = d['val']; Xte, yte, gte = d['test']
sampler = RandomCropSampler(set(gtr))    # extra balanced crops
Xb, yb = sampler.sample(64)
# after prediction: women, p = aggregate_by_woman(window_probs, gte)
```

## Notes
- The signal is **raw**: it has baseline drift and a DC offset. Per-window z-scoring removes the offset only. Add the filters back later by applying them to `SIG[r]` right after cell 2. Everything else stays the same.
- Memory: about 8-9k train windows × 3 × 2400 × 4 B, roughly 250 MB.
- I have not run this code. It follows the channel names in the `.hea` headers, so the first Colab run is its first test.
