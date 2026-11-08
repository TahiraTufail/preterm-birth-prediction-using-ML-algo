# %% [markdown]
# # FYP-2 · Step 1 — Segmentation-based data expansion (leak-free)
#
# **Goal:** turn 300 recordings (38 preterm / 262 term) into thousands of *real-signal*
# training samples for deep learning, without any synthetic data and without leakage.
#
# **Pipeline (same three parallel branches as FYP-1):**
# ```
# Raw EHG (electrodes S1, S2, S3)
#   ├─ Baseline : Butterworth 0.3–3 Hz → 5σ artifact removal
#   ├─ DWT      : db4 L3 soft-threshold → 5σ artifact removal → Butterworth 0.3–3 Hz
#   └─ VMD      : Butterworth 0.3–3 Hz → VMD (K=5, α=2000) → energy-based IMF reconstruction → 5σ
#         ↓   (denoise the FULL recording first, so windows have no filter edge effects)
#   Trim first/last 180 s
#         ↓
#   Assign every WOMAN to one of 5 stratified folds   ← split happens BEFORE windowing
#         ↓
#   Segmentation (same window positions for all 3 branches → fair comparison)
#     • eval windows  : uniform stride, every woman            → used only for val/test
#     • train windows : class-aware stride (more overlap for preterm)
#                       + contraction-centred segments (+ shifted copies for preterm)
#     • random crops  : unlimited on-the-fly crops during training (sampler at the end)
#         ↓
#   get_fold(branch, k) → train windows from train women only, test = eval windows of test women
# ```
#
# **Leakage rules enforced by the code**
# 1. Folds are assigned per woman before any window is cut; a woman's windows never span train and test.
# 2. Expansion (class-aware stride, contraction copies, random crops) is used on train women only.
# 3. Normalisation statistics are computed from the training fold only.
# 4. `gestation` is used only to build the label, never as an input.
# 5. Evaluate per woman (aggregate window predictions) — helper provided at the end.

# %% Install & imports
import sys, subprocess
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'wfdb', 'PyWavelets'], check=True)

import os, glob, json, time, warnings
import numpy as np
import pandas as pd
import wfdb
import pywt
from scipy import signal as sp_signal
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
import matplotlib
if not IN_COLAB:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# %% Paths
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    DATA_DIR = '/content/drive/MyDrive/Colab Notebooks/term-preterm-ehg-database-1.0.1/tpehgdb/'
    OUT_DIR  = '/content/drive/MyDrive/Colab Notebooks/fyp2_segmentation/'
else:
    DATA_DIR = os.environ.get('EHG_DATA_DIR', r'D:\term-preterm-ehg-database-1.0.1\tpehgdb')
    OUT_DIR  = os.environ.get('EHG_OUT_DIR', os.path.join(os.getcwd(), 'fyp2_segmentation'))

assert os.path.isdir(DATA_DIR), f'Folder not found: {DATA_DIR}'
CACHE_DIR = os.path.join(OUT_DIR, 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)
print('DATA_DIR :', DATA_DIR)
print('OUT_DIR  :', OUT_DIR)

# %% Configuration
CFG = {
    'fs': 20,                         # TPEHG sampling rate (Hz)
    'electrodes': ['1', '2', '3'],    # raw channels S1, S2, S3 (the other 9 are pre-filtered copies)
    'branches': ['baseline', 'dwt', 'vmd'],
    'preterm_weeks': 37,              # label = 1 if gestation at delivery < 37 weeks

    # ── preprocessing (identical to FYP-1) ──
    'band': (0.3, 3.0), 'filt_order': 4,
    'artifact_sigma': 5.0,
    'wavelet': 'db4', 'wavelet_level': 3, 'wavelet_mode': 'soft',
    'vmd_K': 5, 'vmd_alpha': 2000, 'vmd_tau': 0.0, 'vmd_DC': 0, 'vmd_init': 1, 'vmd_tol': 1e-7,

    # ── segmentation ──
    'trim_s': 180,                    # drop first/last 3 min (filter transients, set-up artifacts)
    'win_s': 120,                     # window length fed to the DL model
    'eval_stride_s': 60,              # uniform stride for val/test windows (no class-aware tricks)
    'train_stride_s': {0: 60, 1: 15}, # class-aware stride for TRAIN women: term 60 s, preterm 15 s

    # ── contraction-centred segments (detected per recording on the baseline branch) ──
    'env_s': 10,                      # moving-RMS envelope length
    'burst_z': 1.5,                   # threshold = median + z * MAD of the envelope
    'burst_min_s': 30, 'burst_max_s': 300, 'burst_merge_gap_s': 10,
    'contraction_shifts_s': {0: [0], 1: [-20, 0, 20]},  # extra shifted copies for preterm

    'n_folds': 5, 'seed': 42,
    'n_jobs': -1,
}
FS = CFG['fs']
WIN = int(CFG['win_s'] * FS)

# Optional: limit records for a quick test run (e.g. EHG_MAX_RECORDS=20)
MAX_RECORDS = int(os.environ.get('EHG_MAX_RECORDS', '0'))

# %% Load metadata and labels
def parse_header(rec_path):
    """Header comments → dict, e.g. {'Gestation': '35', 'Rectime': '31.3', ...}."""
    meta = {}
    for line in wfdb.rdheader(rec_path).comments:
        parts = line.strip().split(None, 1)
        if len(parts) == 2:
            meta[parts[0]] = parts[1].strip()
    return meta

def to_num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan

rec_paths = sorted(p[:-4] for p in glob.glob(os.path.join(DATA_DIR, '*.hea')))
rows = []
for p in rec_paths:
    h = parse_header(p)
    hdr = wfdb.rdheader(p)
    gest = to_num(h.get('Gestation'))
    rows.append({
        'record': os.path.basename(p),
        'gestation': gest,                                   # label source ONLY — never a feature
        'label': int(gest < CFG['preterm_weeks']),
        'rectime': to_num(h.get('Rectime')),
        'n_samples': hdr.sig_len,
        **{k: h.get(k) for k in ['Age', 'Parity', 'Abortions', 'Weight', 'Hypertension', 'Diabetes',
                                 'Placental_position', 'Bleeding_first_trimester',
                                 'Bleeding_second_trimester', 'Funneling', 'Smoker']},
    })
meta = pd.DataFrame(rows)

if MAX_RECORDS:
    meta = pd.concat([d.head(max(2, int(round(MAX_RECORDS * len(d) / len(meta)))))
                      for _, d in meta.groupby('label')]).reset_index(drop=True)

print(f'Records: {len(meta)}  |  preterm: {meta.label.sum()}  |  term: {(meta.label == 0).sum()}')
print(f'Recording length (min): {meta.n_samples.min() / FS / 60:.1f} – {meta.n_samples.max() / FS / 60:.1f}')

# %% Preprocessing functions (same as FYP-1, applied per electrode)
def bandpass_filter(sig, fs=FS, low=CFG['band'][0], high=CFG['band'][1], order=CFG['filt_order']):
    nyq = fs / 2
    b, a = sp_signal.butter(order, [low / nyq, high / nyq], btype='band')
    return sp_signal.filtfilt(b, a, sig)

def remove_artifacts(sig, threshold=CFG['artifact_sigma']):
    med, std = np.median(sig), np.std(sig)
    out = sig.copy()
    out[np.abs(sig - med) > threshold * std] = med
    return out

def wavelet_denoise(sig):
    coeffs = pywt.wavedec(sig, wavelet=CFG['wavelet'], level=CFG['wavelet_level'])
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    thresh = sigma * np.sqrt(2 * np.log(len(sig)))
    coeffs_t = [coeffs[0]] + [pywt.threshold(c, value=thresh, mode=CFG['wavelet_mode']) for c in coeffs[1:]]
    return pywt.waverec(coeffs_t, wavelet=CFG['wavelet'])[:len(sig)]

def vmd(f, alpha, tau, K, DC, init, tol, n_iter=500):
    """Same algorithm as vmdpy.VMD, but keeps only the current iteration in memory
    (vmdpy stores all 500 iterations, ~2.9 GB for one 30-min recording, which crashes Colab)
    and updates only the positive-frequency half of the spectrum (the negative half stays
    exactly zero in VMD), which halves the run time with identical results."""
    f = np.asarray(f, dtype=np.float64)
    if len(f) % 2:
        f = f[:-1]
    half = len(f) // 2
    f_mirr = np.concatenate([f[:half][::-1], f, f[-half:][::-1]])
    T = len(f_mirr)
    freqs = (np.arange(1, T + 1) / T - 0.5 - 1 / T)[T // 2:]
    f_hat = np.fft.fftshift(np.fft.fft(f_mirr))[T // 2:]          # positive half only

    omega = (0.5 / K) * np.arange(K) if init == 1 else np.zeros(K)
    if DC:
        omega[0] = 0
    lambda_hat = np.zeros_like(f_hat)
    u_hat = np.zeros((K, len(f_hat)), dtype=complex)
    sum_uk = np.zeros_like(f_hat)
    u_diff, n = tol + np.spacing(1), 0

    while u_diff > tol and n < n_iter - 1:
        u_prev = u_hat.copy()
        for k in range(K):
            sum_uk = (u_prev[K - 1] if k == 0 else u_hat[k - 1]) + sum_uk - u_prev[k]
            u_hat[k] = (f_hat - sum_uk - lambda_hat / 2) / (1 + alpha * (freqs - omega[k]) ** 2)
            if not (DC and k == 0):
                p = u_hat[k].real ** 2 + u_hat[k].imag ** 2
                omega[k] = np.dot(freqs, p) / np.sum(p)
        if tau:
            lambda_hat = lambda_hat + tau * (u_hat.sum(axis=0) - f_hat)
        n += 1
        d = u_hat - u_prev
        u_diff = abs(np.spacing(1) + np.sum(d.real ** 2 + d.imag ** 2) / T)

    full = np.zeros((T, K), dtype=complex)
    full[T // 2:] = u_hat.T
    full[np.arange(1, T // 2 + 1)[::-1]] = np.conj(u_hat.T)
    full[0] = np.conj(full[-1])
    u = np.real(np.fft.ifft(np.fft.ifftshift(full, axes=0), axis=0)).T
    return u[:, T // 4:3 * T // 4], omega

def vmd_reconstruct(sig):
    """VMD → keep IMFs whose energy ≥ mean IMF energy → sum."""
    imfs, _ = vmd(sig, CFG['vmd_alpha'], CFG['vmd_tau'], CFG['vmd_K'],
                  CFG['vmd_DC'], CFG['vmd_init'], CFG['vmd_tol'])
    energies = np.sum(imfs ** 2, axis=1)
    selected = imfs[energies >= energies.mean()]
    rec = np.sum(selected if len(selected) else imfs, axis=0)
    if len(rec) < len(sig):                       # vmdpy drops one sample for odd lengths
        rec = np.pad(rec, (0, len(sig) - len(rec)), mode='edge')
    return rec

BRANCH_FUNCS = {
    'baseline': lambda x: remove_artifacts(bandpass_filter(x)),
    'dwt':      lambda x: bandpass_filter(remove_artifacts(wavelet_denoise(x))),
    'vmd':      lambda x: remove_artifacts(vmd_reconstruct(bandpass_filter(x))),
}

def load_electrodes(rec_path):
    """Select S1, S2, S3 by channel NAME (columns 0–2 are S1 raw + two pre-filtered S1 copies)."""
    rec = wfdb.rdrecord(rec_path)
    idx = [rec.sig_name.index(e) for e in CFG['electrodes']]
    return np.nan_to_num(rec.p_signal[:, idx].T.astype(np.float64))   # (3, N)

def preprocess_record(record):
    """Run all branches on one recording; cached to disk so a Colab disconnect can resume."""
    cache = os.path.join(CACHE_DIR, f'{record}.npz')
    if os.path.exists(cache):
        with np.load(cache) as z:
            if all(b in z.files for b in CFG['branches']):
                return record, {b: z[b] for b in CFG['branches']}
    raw = load_electrodes(os.path.join(DATA_DIR, record))
    out = {b: np.stack([BRANCH_FUNCS[b](ch) for ch in raw]).astype(np.float32) for b in CFG['branches']}
    np.savez(cache, **out)
    return record, out

# %% Run preprocessing for all records (parallel, cached)
t0 = time.time()
results = Parallel(n_jobs=CFG['n_jobs'], verbose=5)(delayed(preprocess_record)(r) for r in meta.record)
SIGNALS = {b: {rec: sig[b] for rec, sig in results} for b in CFG['branches']}   # branch → record → (3, N)
print(f'Preprocessed {len(results)} records × {len(CFG["branches"])} branches in {time.time() - t0:.0f} s')

# %% Woman-level stratified folds — assigned BEFORE any windowing
skf = StratifiedKFold(n_splits=CFG['n_folds'], shuffle=True, random_state=CFG['seed'])
meta['fold'] = -1
for k, (_, test_idx) in enumerate(skf.split(meta.record, meta.label)):
    meta.loc[test_idx, 'fold'] = k
print(pd.crosstab(meta.fold, meta.label, margins=True).rename(columns={0: 'term', 1: 'preterm'}))

# %% Segmentation functions
def valid_range(n):
    """Usable sample range after trimming; shrinks the trim for short recordings."""
    trim = int(CFG['trim_s'] * FS)
    if n - 2 * trim < WIN:
        trim = max(0, (n - WIN) // 2)
    return trim, n - trim

def sliding_starts(n, stride_s):
    lo, hi = valid_range(n)
    return np.arange(lo, hi - WIN + 1, int(stride_s * FS))

def envelope(sig):
    """Moving-RMS envelope averaged over electrodes. sig: (3, N)."""
    x = sig - sig.mean(axis=1, keepdims=True)
    k = int(CFG['env_s'] * FS)
    return np.sqrt(np.convolve((x ** 2).mean(axis=0), np.ones(k) / k, mode='same'))

def detect_contractions(sig):
    """Burst detection on one recording's own envelope (no information from other women).
    Returns (bursts, env, thr) where bursts = list of (start, end, peak)."""
    n = sig.shape[1]
    lo, hi = valid_range(n)
    env = envelope(sig)
    seg = env[lo:hi]
    med = np.median(seg)
    mad = 1.4826 * np.median(np.abs(seg - med))
    thr = med + CFG['burst_z'] * mad

    above = np.zeros(n, dtype=bool)
    above[lo:hi] = env[lo:hi] > thr
    edges = np.diff(np.r_[0, above.astype(int), 0])
    runs = list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))

    merged = []
    for s, e in runs:                                   # merge bursts separated by short gaps
        if merged and s - merged[-1][1] < CFG['burst_merge_gap_s'] * FS:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    bursts = []
    for s, e in merged:
        dur = (e - s) / FS
        if CFG['burst_min_s'] <= dur <= CFG['burst_max_s']:
            bursts.append((s, e, s + int(np.argmax(env[s:e]))))
    return bursts, env, thr

def contraction_starts(sig, label):
    """Window starts centred on each burst peak, plus class-dependent shifted copies."""
    n = sig.shape[1]
    lo, hi = valid_range(n)
    bursts, _, _ = detect_contractions(sig)
    starts = []
    for _, _, peak in bursts:
        for shift in CFG['contraction_shifts_s'][label]:
            st = peak - WIN // 2 + int(shift * FS)
            starts.append(int(np.clip(st, lo, hi - WIN)))
    return sorted(set(starts)), len(bursts)

# %% Build the window table (positions shared by all three branches)
win_rows = []
for r in meta.itertuples():
    sig_ref = SIGNALS['baseline'][r.record] if 'baseline' in SIGNALS else next(iter(SIGNALS.values()))[r.record]
    n = sig_ref.shape[1]
    base = {'record': r.record, 'label': r.label, 'fold': r.fold}

    # 1) eval windows: uniform stride, every woman (only ever used as val/test)
    for st in sliding_starts(n, CFG['eval_stride_s']):
        win_rows.append({**base, 'start': int(st), 'set': 'eval', 'source': 'sliding'})

    # 2) train windows: class-aware stride (only used when this woman is in the training folds)
    for st in sliding_starts(n, CFG['train_stride_s'][r.label]):
        win_rows.append({**base, 'start': int(st), 'set': 'train', 'source': 'sliding'})

    # 3) train windows: contraction-centred segments
    c_starts, n_bursts = contraction_starts(sig_ref, r.label)
    meta.loc[meta.record == r.record, 'n_contractions'] = n_bursts
    for st in c_starts:
        win_rows.append({**base, 'start': int(st), 'set': 'train', 'source': 'contraction'})

windows = pd.DataFrame(win_rows)
windows['end'] = windows['start'] + WIN

# %% Expansion summary
def summarise(df):
    g = df.groupby('label')
    return pd.DataFrame({'women': g.record.nunique(), 'windows': g.size(),
                         'per_woman': (g.size() / g.record.nunique()).round(1)})

print('Original data      : 300 recordings → 1 sample per woman' if len(meta) == 300 else
      f'Original data      : {len(meta)} recordings')
print('\nEVAL windows (uniform stride — for val/test):')
print(summarise(windows[windows.set == 'eval']).rename(index={0: 'term', 1: 'preterm'}))
print('\nTRAIN windows (class-aware sliding + contraction-centred):')
print(summarise(windows[windows.set == 'train']).rename(index={0: 'term', 1: 'preterm'}))
print('\nTRAIN windows by source:')
print(pd.crosstab(windows[windows.set == 'train'].source,
                  windows[windows.set == 'train'].label).rename(columns={0: 'term', 1: 'preterm'}))
print(f'\nContractions detected per woman: mean {meta.n_contractions.mean():.1f}, '
      f'zero in {(meta.n_contractions == 0).sum()} recordings')

# %% Leak-free fold loader
def materialize(df, branch):
    sigs = SIGNALS[branch]
    X = np.stack([sigs[r][:, s:e] for r, s, e in zip(df.record, df.start, df.end)]).astype(np.float32)
    return X, df.label.values.astype(np.int64), df.record.values

def get_fold(branch, test_fold, use_val=True, norm='fold'):
    """Leak-free arrays for one outer fold.

    train : 'train' windows of women NOT in the test (or val) fold   ← expansion applied
    val   : 'eval'  windows of women in fold (test_fold + 1)          ← no expansion
    test  : 'eval'  windows of women in test_fold                     ← no expansion
    norm  : 'fold'   → per-electrode mean/std from TRAIN windows only
            'window' → z-score each window on its own
            None     → leave as is
    Shapes: X (n_windows, 3, WIN), y (n_windows,), groups = record id per window.
    """
    val_fold = (test_fold + 1) % CFG['n_folds'] if use_val else None
    held_out = {test_fold} | ({val_fold} if use_val else set())

    tr = windows[(windows.set == 'train') & (~windows.fold.isin(held_out))]
    te = windows[(windows.set == 'eval') & (windows.fold == test_fold)]
    va = windows[(windows.set == 'eval') & (windows.fold == val_fold)] if use_val else None

    out = {'train': materialize(tr, branch), 'test': materialize(te, branch)}
    if use_val:
        out['val'] = materialize(va, branch)

    # ── leakage checks ──
    tr_w, te_w = set(out['train'][2]), set(out['test'][2])
    assert tr_w.isdisjoint(te_w), 'LEAK: a woman is in both train and test'
    if use_val:
        va_w = set(out['val'][2])
        assert tr_w.isdisjoint(va_w) and va_w.isdisjoint(te_w), 'LEAK: val overlaps train/test'

    if norm == 'fold':
        Xtr = out['train'][0]
        mu = Xtr.mean(axis=(0, 2), keepdims=True)
        sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-8
        for key in out:
            X, y, g = out[key]
            out[key] = ((X - mu) / sd, y, g)
        out['norm_stats'] = (mu, sd)            # reuse for random crops
    elif norm == 'window':
        for key in out:
            X, y, g = out[key]
            out[key] = ((X - X.mean(2, keepdims=True)) / (X.std(2, keepdims=True) + 1e-8), y, g)
    return out

def electrodes_as_samples(X, y, g):
    """Optional: treat each electrode as its own single-channel sample (apply AFTER get_fold)."""
    n, c, L = X.shape
    return X.reshape(n * c, 1, L), np.repeat(y, c), np.repeat(g, c), np.tile(np.arange(c), n)

def aggregate_by_woman(probs, groups, how='mean'):
    """Window probabilities → one probability per woman (use this for AUC / sensitivity)."""
    df = pd.DataFrame({'record': groups, 'p': probs})
    agg = df.groupby('record').p.agg(how)
    return agg.index.values, agg.values

# %% Quick check of one fold
fold0 = get_fold('baseline', test_fold=0)
for split in ['train', 'val', 'test']:
    X, y, g = fold0[split]
    print(f'{split:5s}: X {X.shape} | windows preterm {int(y.sum()):5d} / term {int((y == 0).sum()):5d} '
          f'| women {len(set(g)):3d}')

# %% Random-crop sampler (on-the-fly expansion during training, train women only)
class RandomCropSampler:
    """Balanced random crops from the continuous signals of TRAIN women.

    Every call returns new crop positions, so the model almost never sees the same input twice.
    Use together with get_fold(...)['val'/'test'] for evaluation.
    """
    def __init__(self, branch, train_records, norm_stats=None, balanced=True, seed=0):
        self.sigs = SIGNALS[branch]
        m = meta.set_index('record').loc[list(train_records)]
        self.by_class = {c: m.index[m.label == c].tolist() for c in (0, 1)}
        self.ranges = {r: valid_range(self.sigs[r].shape[1]) for r in m.index}
        self.norm = norm_stats
        self.balanced = balanced
        self.rng = np.random.default_rng(seed)
        self.all = m.index.tolist()
        self.labels = m.label.to_dict()

    def sample(self, batch_size):
        X = np.empty((batch_size, len(CFG['electrodes']), WIN), np.float32)
        y = np.empty(batch_size, np.int64)
        for i in range(batch_size):
            if self.balanced:
                c = int(self.rng.random() < 0.5)
                rec = self.by_class[c][self.rng.integers(len(self.by_class[c]))]
            else:
                rec = self.all[self.rng.integers(len(self.all))]
            lo, hi = self.ranges[rec]
            st = self.rng.integers(lo, hi - WIN + 1)
            X[i] = self.sigs[rec][:, st:st + WIN]
            y[i] = self.labels[rec]
        if self.norm is not None:
            mu, sd = self.norm
            X = (X - mu) / sd
        return X, y

try:
    import torch
    from torch.utils.data import Dataset

    class EHGRandomCropDataset(Dataset):
        """PyTorch wrapper: each epoch draws `epoch_size` fresh balanced crops."""
        def __init__(self, sampler, epoch_size=4096):
            self.sampler, self.epoch_size = sampler, epoch_size
        def __len__(self):
            return self.epoch_size
        def __getitem__(self, idx):
            X, y = self.sampler.sample(1)
            return torch.from_numpy(X[0]), torch.tensor(y[0], dtype=torch.float32)
except ImportError:
    pass

train_women = sorted(set(fold0['train'][2]))
sampler = RandomCropSampler('baseline', train_women, norm_stats=fold0['norm_stats'], seed=CFG['seed'])
Xb, yb = sampler.sample(64)
print(f'Random-crop batch: {Xb.shape}, preterm fraction {yb.mean():.2f}')

# %% Visualise segmentation for one preterm and one term woman
fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=False)
for ax, lab in zip(axes, [1, 0]):
    rec = meta[meta.label == lab].sort_values('n_contractions', ascending=False).record.iloc[0]
    sig = SIGNALS['baseline'][rec]
    bursts, env, thr = detect_contractions(sig)
    t = np.arange(sig.shape[1]) / FS / 60
    ax.plot(t, sig[0], lw=0.4, color='0.6', label='S1 (baseline branch)')
    scale = np.abs(sig[0]).max() / env.max()
    ax.plot(t, env * scale, color='C0', lw=1.2, label='envelope')
    ax.axhline(thr * scale, color='C0', ls='--', lw=0.8, label='burst threshold')
    for s, e, _ in bursts:
        ax.axvspan(s / FS / 60, e / FS / 60, color='C3', alpha=0.15)
    lo, hi = valid_range(sig.shape[1])
    ax.axvspan(0, lo / FS / 60, color='k', alpha=0.08)
    ax.axvspan(hi / FS / 60, t[-1], color='k', alpha=0.08)
    n_tr = ((windows.record == rec) & (windows.set == 'train')).sum()
    ax.set_title(f'{rec} — {"PRETERM" if lab else "TERM"} | {len(bursts)} contractions (red) | '
                 f'{n_tr} train windows from 1 recording (grey = trimmed)')
    ax.set_xlabel('time (min)')
    ax.legend(loc='upper right', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'segmentation_example.png'), dpi=130)
plt.show()

fig, ax = plt.subplots(figsize=(7, 4))
counts = pd.DataFrame({
    'recordings': meta.groupby('label').size(),
    'train windows': windows[windows.set == 'train'].groupby('label').size(),
}).rename(index={0: 'term', 1: 'preterm'})
counts.T.plot.bar(ax=ax, rot=0, color=['C0', 'C3'])
ax.set_ylabel('samples'); ax.set_title('Samples before vs after segmentation (all folds)')
for c in ax.containers:
    ax.bar_label(c, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'expansion_counts.png'), dpi=130)
plt.show()

# %% Save everything for the deep-learning notebook
meta.drop(columns=['gestation']).to_csv(os.path.join(OUT_DIR, 'meta.csv'), index=False)  # label kept, gestation dropped
windows.to_csv(os.path.join(OUT_DIR, 'windows.csv'), index=False)
for b in CFG['branches']:
    np.savez(os.path.join(OUT_DIR, f'signals_{b}.npz'), **SIGNALS[b])
with open(os.path.join(OUT_DIR, 'config.json'), 'w') as f:
    json.dump({k: v if not isinstance(v, dict) else {str(kk): vv for kk, vv in v.items()}
               for k, v in CFG.items()}, f, indent=2, default=list)
print('Saved:', sorted(os.listdir(OUT_DIR)))

# %% [markdown]
# ### How to use this in the deep-learning notebook
# ```python
# for k in range(5):                                   # outer folds = women
#     d = get_fold('vmd', test_fold=k)                 # or 'baseline' / 'dwt'
#     sampler = RandomCropSampler('vmd', set(d['train'][2]), d['norm_stats'])
#     # train on d['train'] windows and/or sampler.sample(batch) crops
#     # early-stop on d['val'], then predict d['test'] windows:
#     # women, p = aggregate_by_woman(window_probs, d['test'][2]);  AUC on (women, p)
# ```
# Re-load later without re-running VMD: `np.load(OUT_DIR + 'signals_vmd.npz')` + `windows.csv`.
