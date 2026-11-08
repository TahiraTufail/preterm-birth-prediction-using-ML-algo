import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE, SVMSMOTE, ADASYN
from collections import Counter
import shap
import warnings
warnings.filterwarnings('ignore')

import os
from google.colab import drive
drive.mount('/content/drive')

ROOT = "/content/drive/MyDrive/Colab Notebooks"
PROJECT_FOLDER = None
for f in os.listdir(ROOT):
    if "term-preterm-ehg" in f.lower():
        PROJECT_FOLDER = f
        break
if PROJECT_FOLDER is None:
    raise FileNotFoundError("EHG dataset folder not found")
BASE_DIR = os.path.join(ROOT, PROJECT_FOLDER)

# ═══════════════════════════════════════════════════════════════════
# 1. LOAD BASE DATA
# ═══════════════════════════════════════════════════════════════════
file_path = os.path.join(BASE_DIR, "ehg_preprocessed_0_3_4_0hz.csv")
df = pd.read_csv(file_path)

drop_cols = [c for c in ['Record', 'Rec_time', 'rectime',
                          'Gestation', 'gestation', 'rec_num',
                          'Group', 'Early'] if c in df.columns]
X_base = df.drop(columns=['label'] + drop_cols).select_dtypes(include=[np.number])
y      = df['label']

# ═══════════════════════════════════════════════════════════════════
# 2. ADD STATISTICAL FEATURES (variance, skewness, kurtosis, ZCR)
#    Computed per channel from the raw signal columns already in CSV
#    ch1, ch2, ch3 — each has rms, fmed, fpeak, samp_en
#    We derive stats across those 4 features per channel as a proxy
# ═══════════════════════════════════════════════════════════════════

def add_statistical_features(X_df):
    """
    For each of the 3 EHG channels, compute:
      - variance, skewness, kurtosis across existing channel features
      - zero_crossing_rate proxy (sign changes in [rms, fmed, fpeak, samp_en])

    These capture distributional shape of feature vectors per channel,
    adding 4 stats × 3 channels = 12 new features.
    """
    new_feats = {}

    for ch in ['ch1', 'ch2', 'ch3']:
        # Collect all columns belonging to this channel
        ch_cols = [c for c in X_df.columns if c.startswith(ch + '_')]

        if len(ch_cols) == 0:
            continue

        ch_data = X_df[ch_cols].values   # shape: (n_samples, n_ch_features)

        # Variance across channel features per sample
        new_feats[f'{ch}_variance']  = np.var(ch_data,  axis=1)

        # Skewness across channel features per sample
        new_feats[f'{ch}_skewness']  = skew(ch_data,    axis=1)

        # Kurtosis across channel features per sample
        new_feats[f'{ch}_kurtosis']  = kurtosis(ch_data, axis=1)

        # Zero-crossing rate: how often sign changes across channel features
        # (proxy — meaningful when features have natural pos/neg variation)
        signs   = np.sign(ch_data)
        zcr     = np.sum(np.abs(np.diff(signs, axis=1)), axis=1) / (ch_data.shape[1] - 1)
        new_feats[f'{ch}_zcr']       = zcr

    new_df = pd.DataFrame(new_feats, index=X_df.index)
    return pd.concat([X_df, new_df], axis=1)


X = add_statistical_features(X_base)

print("=" * 65)
print("DATASET OVERVIEW")
print("=" * 65)
print(f"Shape                    : {df.shape}")
print(f"Base features            : {X_base.shape[1]}")
print(f"After statistical feats  : {X.shape[1]}  (+{X.shape[1]-X_base.shape[1]} new)")
print(f"Class distribution:")
print(f"  Non-Preterm (0) : {(y==0).sum()}")
print(f"  Preterm     (1) : {(y==1).sum()}")
print(f"\nNew statistical features added:")
new_cols = [c for c in X.columns if c not in X_base.columns]
for c in new_cols:
    print(f"  {c}")

# ═══════════════════════════════════════════════════════════════════
# 3. CONFIG — choose oversampler & classification threshold
# ═══════════════════════════════════════════════════════════════════

# ── Oversampler options ────────────────────────────────────────────
#   'smote'    → plain SMOTE         (original baseline)
#   'svmsmote' → SVM-boundary SMOTE  (better minority boundaries)
#   'adasyn'   → adaptive synthesis  (focuses on hard samples)

OVERSAMPLER = 'svmsmote'   # ← change to 'smote' / 'adasyn' to compare

def get_oversampler(name):
    if name == 'smote':
        return SMOTE(random_state=42)
    elif name == 'svmsmote':
        return SVMSMOTE(random_state=42)
    elif name == 'adasyn':
        return ADASYN(random_state=42, n_neighbors=5)
    else:
        raise ValueError(f"Unknown oversampler: {name}")

# ── Classification threshold ───────────────────────────────────────
#   Default = 0.5  (balanced precision/recall)
#   Lower   = 0.3  (better sensitivity, more false positives)
#   Clinical use → prefer higher sensitivity for preterm detection

THRESHOLD = 0.3    # ← change to 0.5 to use default

print(f"\nConfiguration:")
print(f"  Oversampler : {OVERSAMPLER}")
print(f"  Threshold   : {THRESHOLD}  (default=0.5)")

# ═══════════════════════════════════════════════════════════════════
# 4. SHAP FEATURE SELECTOR (inside fold)
# ═══════════════════════════════════════════════════════════════════
N_FEATURES = 10

def get_shap_top_features(shap_selector, X_train_sm, y_train_sm,
                           feature_names, n_features, use_kernel=False):
    """
    Fit shap_selector on training data only → return top-n feature
    indices & names by mean |SHAP| value.
    """
    shap_selector.fit(X_train_sm, y_train_sm)

    if use_kernel:
        background = shap.kmeans(X_train_sm, 20)
        explainer  = shap.KernelExplainer(
            lambda x: shap_selector.predict_proba(x)[:, 1], background
        )
        rng        = np.random.RandomState(42)
        sample_idx = rng.choice(len(X_train_sm),
                                size=min(150, len(X_train_sm)),
                                replace=False)
        shap_vals  = explainer.shap_values(X_train_sm[sample_idx])
    else:
        explainer    = shap.TreeExplainer(shap_selector)
        shap_output  = explainer.shap_values(X_train_sm)
        if isinstance(shap_output, list):
            shap_vals = shap_output[1]
        elif shap_output.ndim == 3:
            shap_vals = shap_output[:, :, 1]
        else:
            shap_vals = shap_output

    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    top_idx       = np.argsort(mean_abs_shap)[::-1][:n_features]
    top_cols      = [feature_names[i] for i in top_idx]
    return top_idx, top_cols

# ═══════════════════════════════════════════════════════════════════
# 5. CROSS-VALIDATED EVALUATION
#    Scale + Oversample + SHAP + Threshold — ALL inside fold
# ═══════════════════════════════════════════════════════════════════
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_pipeline(shap_selector, final_model, label,
                       use_kernel=False,
                       oversampler_name=OVERSAMPLER,
                       threshold=THRESHOLD):

    accs, aucs, f1s, cms   = [], [], [], []
    fold_features          = []

    X_arr         = X.values
    y_arr         = y.values
    feature_names = X.columns.tolist()

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_arr, y_arr), 1):
        print(f"  Fold {fold}/5 ...", end="\r")

        X_train_raw, X_test_raw = X_arr[train_idx], X_arr[test_idx]
        y_train,     y_test     = y_arr[train_idx], y_arr[test_idx]

        # ── Step A: Scale INSIDE fold ──────────────────────────────
        scaler         = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled  = scaler.transform(X_test_raw)

        # ── Step B: Oversample INSIDE fold (train only) ────────────
        oversampler            = get_oversampler(oversampler_name)
        X_train_sm, y_train_sm = oversampler.fit_resample(
                                     X_train_scaled, y_train)

        # ── Step C: SHAP feature selection INSIDE fold ─────────────
        top_idx, top_cols = get_shap_top_features(
            shap_selector, X_train_sm, y_train_sm,
            feature_names, N_FEATURES, use_kernel=use_kernel
        )
        fold_features.append(top_cols)

        X_train_sel = X_train_sm[:, top_idx]
        X_test_sel  = X_test_scaled[:, top_idx]   # no refit on test

        # ── Step D: Train final model ──────────────────────────────
        final_model.fit(X_train_sel, y_train_sm)
        y_prob = final_model.predict_proba(X_test_sel)[:, 1]

        # ── Step E: Apply custom threshold ────────────────────────
        y_pred = (y_prob >= threshold).astype(int)

        accs.append((y_pred == y_test).mean())
        aucs.append(roc_auc_score(y_test, y_prob))   # AUC uses probabilities
        f1s.append(f1_score(y_test, y_pred, zero_division=0))
        cms.append(confusion_matrix(y_test, y_pred))

    # ── Print results ──────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"  {label}")
    print(f"  Oversampler: {oversampler_name.upper()}  |  Threshold: {threshold}")
    print(f"{'=' * 65}")
    print(f"  Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  AUC-ROC  : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"  F1-Score : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    total_cm       = np.sum(cms, axis=0)
    tn, fp, fn, tp = total_cm.ravel()
    sensitivity    = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity    = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"\n  Confusion Matrix (summed across 5 folds):")
    print(f"                  Predicted 0   Predicted 1")
    print(f"  Actual 0  :     {tn:>9}   {fp:>9}")
    print(f"  Actual 1  :     {fn:>9}   {tp:>9}")
    print(f"\n  Sensitivity (Recall for Preterm) : {sensitivity:.4f}")
    print(f"  Specificity (Recall for Non-PT)  : {specificity:.4f}")

    # Feature stability
    all_selected = [f for fold in fold_features for f in fold]
    freq         = Counter(all_selected).most_common()
    print(f"\n  Feature selection frequency across 5 folds:")
    for feat, count in freq:
        bar = "█" * count + "░" * (5 - count)
        print(f"    {feat:<35} {bar}  {count}/5")

    return np.mean(accs), np.mean(aucs), np.mean(f1s)


# ═══════════════════════════════════════════════════════════════════
# 6. RUN PIPELINES
# ═══════════════════════════════════════════════════════════════════

# ── SHAP + SVM ─────────────────────────────────────────────────────
print(f"\nRunning SHAP + SVM  [{OVERSAMPLER.upper()} | threshold={THRESHOLD}]...")
svm_acc, svm_auc, svm_f1 = evaluate_pipeline(
    shap_selector      = SVC(kernel='rbf', C=1, gamma='scale',
                             class_weight='balanced', random_state=42,
                             probability=True),
    final_model        = SVC(kernel='rbf', C=1, gamma='scale',
                             class_weight='balanced', random_state=42,
                             probability=True),
    label              = "SHAP + SVM  (leak-free)",
    use_kernel         = True,
    oversampler_name   = OVERSAMPLER,
    threshold          = THRESHOLD
)

# ── SHAP + RFC ─────────────────────────────────────────────────────
print(f"\nRunning SHAP + RFC  [{OVERSAMPLER.upper()} | threshold={THRESHOLD}]...")
rfc_acc, rfc_auc, rfc_f1 = evaluate_pipeline(
    shap_selector      = RandomForestClassifier(n_estimators=100,
                             class_weight='balanced', random_state=42),
    final_model        = RandomForestClassifier(n_estimators=100,
                             class_weight='balanced', random_state=42),
    label              = "SHAP + RFC  (leak-free)",
    use_kernel         = False,
    oversampler_name   = OVERSAMPLER,
    threshold          = THRESHOLD
)

# ═══════════════════════════════════════════════════════════════════
# 7. FINAL COMPARISON
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 65}")
print(f"  FINAL MODEL COMPARISON")
print(f"  [{OVERSAMPLER.upper()} oversampler | threshold = {THRESHOLD}]")
print(f"{'=' * 65}")
print(f"  {'Metric':<15} {'SHAP+SVM':>12} {'SHAP+RFC':>12}")
print(f"  {'-' * 42}")
print(f"  {'Accuracy':<15} {svm_acc:>12.4f} {rfc_acc:>12.4f}")
print(f"  {'AUC-ROC':<15} {svm_auc:>12.4f} {rfc_auc:>12.4f}")
print(f"  {'F1-Score':<15} {svm_f1:>12.4f} {rfc_f1:>12.4f}")
print(f"{'=' * 65}")

best = "SHAP+SVM" if svm_auc > rfc_auc else "SHAP+RFC"
print(f"\n  ✅ Best model (by AUC-ROC): {best}")
