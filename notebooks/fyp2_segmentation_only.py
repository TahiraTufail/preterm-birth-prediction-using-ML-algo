# %% [markdown]
# # FYP-2 · Segmentation only (no filters, no VMD/DWT)
# Raw TPEHG electrodes S1, S2, S3 -> trim -> woman-level folds -> windows (class-aware stride)
# + balanced random-crop sampler. Run in Google Colab.

# %% 1. Setup
!pip install -q wfdb

import os, glob
import numpy as np
import pandas as pd
import wfdb
from sklearn.model_selection import StratifiedKFold
from google.colab import drive
drive.mount('/content/drive')

DATA_DIR = '/content/drive/MyDrive/Colab Notebooks/term-preterm-ehg-database-1.0.1/tpehgdb/'
OUT_DIR  = '/content/drive/MyDrive/Colab Notebooks/fyp2_segmentation/'
os.makedirs(OUT_DIR, exist_ok=True)

FS         = 20            # Hz
TRIM_S     = 180           # drop first/last 3 min
WIN_S      = 120           # window length (s)
EVAL_STRIDE_S  = 60                  # val/test windows: same for every woman
TRAIN_STRIDE_S = {0: 60, 1: 15}      # train windows: term 60 s, preterm 15 s (more overlap)
N_FOLDS, SEED  = 5, 42
WIN = WIN_S * FS

# %% 2. Load raw signals + labels  (no filtering of any kind)
def header_info(path):
    d = {}
    for line in wfdb.rdheader(path).comments:
        p = line.strip().split(None, 1)
        if len(p) == 2:
            d[p[0]] = p[1]
    return d

records = sorted(os.path.basename(p)[:-4] for p in glob.glob(DATA_DIR + '*.hea'))
SIG, rows = {}, []
for r in records:
    path = DATA_DIR + r
    rec  = wfdb.rdrecord(path)
    idx  = [rec.sig_name.index(c) for c in ('1', '2', '3')]        # raw S1, S2, S3 only
    SIG[r] = np.nan_to_num(rec.p_signal[:, idx].T).astype(np.float32)   # (3, N)
    gest = float(header_info(path)['Gestation'])
    rows.append({'record': r, 'label': int(gest < 37)})            # gestation used for the label only
meta = pd.DataFrame(rows)
print(f'{len(meta)} records | preterm {meta.label.sum()} | term {(meta.label == 0).sum()}')

# %% 3. Woman-level folds  (assigned BEFORE any windowing)
meta['fold'] = -1
skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
for k, (_, te) in enumerate(skf.split(meta.record, meta.label)):
    meta.loc[te, 'fold'] = k

# %% 4. Segmentation -> window table
def valid_range(n):
    trim = TRIM_S * FS
    if n - 2 * trim < WIN:                       # very short recording: trim less
        trim = max(0, (n - WIN) // 2)
    return trim, n - trim

def starts(n, stride_s):
    lo, hi = valid_range(n)
    return range(lo, hi - WIN + 1, stride_s * FS)

rows = []
for r in meta.itertuples():
    n = SIG[r.record].shape[1]
    for st in starts(n, EVAL_STRIDE_S):                       # used only for val/test
        rows.append((r.record, r.label, r.fold, st, 'eval'))
    for st in starts(n, TRAIN_STRIDE_S[r.label]):             # used only if woman is in train folds
        rows.append((r.record, r.label, r.fold, st, 'train'))
windows = pd.DataFrame(rows, columns=['record', 'label', 'fold', 'start', 'set'])
windows['end'] = windows.start + WIN

tr = windows[windows.set == 'train']
print('\nSamples before: 1 per woman  ->  after (train windows, all folds):')
print(pd.DataFrame({'women': tr.groupby('label').record.nunique(),
                    'windows': tr.groupby('label').size()}).rename(index={0: 'term', 1: 'preterm'}))

# %% 5. Leak-free fold loader
def materialize(df):
    X = np.stack([SIG[r][:, s:e] for r, s, e in zip(df.record, df.start, df.end)])
    return X, df.label.values.astype(np.int64), df.record.values

def get_fold(test_fold, norm='window'):
    """train: expanded windows of train women | val: eval windows of fold+1 | test: eval windows of test_fold.
    norm='window' z-scores every window on its own (recommended for raw, unfiltered signal)."""
    val_fold = (test_fold + 1) % N_FOLDS
    out = {
        'train': materialize(windows[(windows.set == 'train') & ~windows.fold.isin([test_fold, val_fold])]),
        'val':   materialize(windows[(windows.set == 'eval') & (windows.fold == val_fold)]),
        'test':  materialize(windows[(windows.set == 'eval') & (windows.fold == test_fold)]),
    }
    a, b, c = (set(out[k][2]) for k in ('train', 'val', 'test'))
    assert a.isdisjoint(b) and a.isdisjoint(c) and b.isdisjoint(c), 'LEAK: a woman is in two splits'
    if norm == 'window':
        for k, (X, y, g) in out.items():
            X = X - X.mean(2, keepdims=True)
            out[k] = (X / (X.std(2, keepdims=True) + 1e-8), y, g)
    return out

# %% 6. Balanced random-crop sampler (extra on-the-fly samples, TRAIN women only)
class RandomCropSampler:
    def __init__(self, train_records, seed=0):
        m = meta.set_index('record').loc[list(train_records)]
        self.cls = {c: m.index[m.label == c].tolist() for c in (0, 1)}
        self.lab, self.rng = m.label.to_dict(), np.random.default_rng(seed)
    def sample(self, batch_size):                    # ~50% preterm per batch
        X, y = [], []
        for _ in range(batch_size):
            c   = int(self.rng.random() < 0.5)
            rec = self.cls[c][self.rng.integers(len(self.cls[c]))]
            lo, hi = valid_range(SIG[rec].shape[1])
            st  = self.rng.integers(lo, hi - WIN + 1)
            x   = SIG[rec][:, st:st + WIN]
            x   = x - x.mean(1, keepdims=True)
            X.append(x / (x.std(1, keepdims=True) + 1e-8)); y.append(self.lab[rec])
        return np.stack(X).astype(np.float32), np.array(y)

def aggregate_by_woman(probs, groups, how='mean'):
    """Window probabilities -> one probability per woman. Compute AUC / sensitivity on THIS."""
    a = pd.DataFrame({'record': groups, 'p': probs}).groupby('record').p.agg(how)
    return a.index.values, a.values

# %% 7. Check fold 0
d = get_fold(0)
for k in ('train', 'val', 'test'):
    X, y, g = d[k]
    print(f'{k:5s} X{X.shape} | preterm windows {int(y.sum()):5d} | term windows {int((y == 0).sum()):5d} | women {len(set(g))}')
Xb, yb = RandomCropSampler(set(d['train'][2])).sample(64)
print('random-crop batch', Xb.shape, '| preterm fraction', yb.mean())

# %% 8. Save (optional)
meta.to_csv(OUT_DIR + 'meta.csv', index=False)
windows.to_csv(OUT_DIR + 'windows.csv', index=False)
print('saved meta.csv and windows.csv to', OUT_DIR)
