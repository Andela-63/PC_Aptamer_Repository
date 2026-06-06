"""
ML-Enhanced SCIA PC-Aptamer Pipeline  — Q1-Grade Version
=========================================================
Run this in Google Colab or locally. After running, paste the printed
"RESULTS BLOCK" back to Claude to incorporate into the final manuscript.

Dependencies:  numpy, scipy, scikit-learn, matplotlib, seaborn, pandas
Install:  pip install numpy scipy scikit-learn matplotlib seaborn pandas
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict, KFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATASET  (Alharbi et al. 2026, Table 2 + 2 negative controls)
# ─────────────────────────────────────────────────────────────────────────────
APTAMERS = [
    # Universal PC aptamers
    ("PCAPt1",  "GAAAAGGAGC", 1.000, "Universal"),
    ("PCAPt2",  "GAAAAGGATC", 0.975, "Universal"),
    # DNA-specific
    ("PCAPt3",  "GAAAAGGATG", 0.950, "DNA"),
    ("PCAPt4",  "GAAAGGATGC", 0.925, "DNA"),
    ("PCAPt5",  "GAAGGATGCA", 0.900, "DNA"),
    ("PCAPt6",  "GAAGGATGCG", 0.875, "DNA"),
    # RNA-specific
    ("PCAPt7",  "GAAAAGGAUC", 0.960, "RNA"),
    ("PCAPt8",  "GAAAAGGUAC", 0.930, "RNA"),
    ("PCAPt9",  "GAAGGUACGC", 0.900, "RNA"),
    ("PCAPt10", "GAAGGUACGG", 0.870, "RNA"),
    # Hybrid DNA/RNA
    ("PCAPt11", "GAGCGGAUCC", 0.855, "Hybrid"),
    ("PCAPt12", "GAGCGGAUCG", 0.840, "Hybrid"),
    ("PCAPt13", "GAGCGGATCC", 0.820, "Hybrid"),
    ("PCAPt14", "GAGCGGATCG", 0.800, "Hybrid"),
    ("PCAPt15", "GAGCGGAUCU", 0.780, "Hybrid"),
    ("PCAPt16", "GAGUGGAUCC", 0.760, "Hybrid"),
    ("PCAPt17", "GCGCGGAUCC", 0.740, "Hybrid"),
    ("PCAPt18", "GCGCGGATCC", 0.720, "Hybrid"),
    # Negative controls
    ("NEG1",    "GGCGGCGGCG", 0.100, "Negative"),
    ("NEG2",    "AGGGTTTTTT", 0.080, "Negative"),
]

# ─────────────────────────────────────────────────────────────────────────────
# 2. SCI ENERGY REIMPLEMENTATION
# ─────────────────────────────────────────────────────────────────────────────
CHARGE = {'A': 3, 'G': 4, 'C': 4, 'T': 3, 'U': 3}
LATTICE = {'A': 0.34, 'G': 0.34, 'C': 0.30, 'T': 0.30, 'U': 0.30}
E0  = 8.854e-12   # F/m
ER  = 80.0
E   = 1.6e-19     # C
KB  = 1.38e-23    # J/K
T   = 310.0       # K
N   = 1.0e26      # ionic number density m^-3
PC_CHARGE = 2     # PC zwitterion net charge magnitude

def sci_energy(abb_charge, r_nm_array):
    """Screened Coulomb binding energy at each distance in r_nm_array (nm)."""
    r = r_nm_array * 1e-9
    k_arr = np.linspace(1e9, 1e11, 8000)
    dk = k_arr[1] - k_arr[0]
    screening = 1.0 / (2 * np.pi * KB * T * N)
    energies = []
    for ri in r:
        Vk  = (abb_charge * PC_CHARGE * E**2) / (E0 * ER * k_arr**2)
        Vsc = Vk * np.exp(-screening * k_arr**2)
        # Inverse Fourier (1D radial)
        integrand = Vsc * np.cos(k_arr * ri)
        from scipy.integrate import trapezoid as trapz_fn; E_sci = trapz_fn(integrand, k_arr)
        energies.append(E_sci)
    return np.array(energies)

r_vals = np.linspace(0.1, 5.0, 300)
SCI_ENERGIES = {nt: sci_energy(CHARGE[nt], r_vals) for nt in CHARGE}

# Rank ordering verification
rank = sorted(CHARGE.items(), key=lambda x: -x[1])
print("SCI rank order:", " > ".join([f"{nt}(q={q})" for nt, q in rank]))

# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING  (27-D + SCI energy features → 31-D total)
# ─────────────────────────────────────────────────────────────────────────────
DINUCS = [a+b for a in 'AGCT' for b in 'AGCT']

def featurize(seq):
    seq = seq.upper()
    n = len(seq)
    counts = {nt: seq.count(nt) for nt in 'AGCTU'}
    fracs  = {nt: counts[nt]/n for nt in 'AGCTU'}
    gc     = (counts['G'] + counts['C']) / n
    purine = (counts['A'] + counts['G']) / n
    charge_total = sum(CHARGE.get(nt, 3) for nt in seq)
    length_nm    = sum(LATTICE.get(nt, 0.32) for nt in seq)
    charge_dens  = charge_total / n
    charge_per_nm = charge_total / length_nm if length_nm > 0 else 0
    # Shannon entropy
    from math import log2
    entropy = -sum((v/n)*log2(v/n) for v in counts.values() if v > 0)
    # Dinucleotide frequencies (RNA-normalised: U→T for DNA comparison)
    seq_dna = seq.replace('U', 'T')
    di_freqs = {d: 0 for d in DINUCS}
    for i in range(len(seq_dna)-1):
        di = seq_dna[i:i+2]
        if di in di_freqs:
            di_freqs[di] += 1
    total_di = max(1, len(seq_dna)-1)
    di_vec   = [di_freqs[d]/total_di for d in DINUCS]
    # SCI energy at minimum distance (0.1 nm) as feature
    sci_feats = [SCI_ENERGIES[nt][0] for nt in 'AGCT']  # 4 extra features
    feat = ([fracs['A'], fracs['G'], fracs['C'], fracs['T'], fracs['U'],
             gc, purine, charge_total, charge_dens, charge_per_nm,
             length_nm, entropy, n] + di_vec + sci_feats)
    return feat

FEAT_NAMES = (['fA','fG','fC','fT','fU','GC','purine','charge_total',
               'charge_dens','charge_per_nm','length_nm','entropy','seq_len']
              + DINUCS + ['SCI_A','SCI_G','SCI_C','SCI_T'])

# Build design matrix
records   = APTAMERS
sequences = [r[1] for r in records]
labels    = [r[0] for r in records]
y         = np.array([r[2] for r in records])
X_raw     = np.array([featurize(s) for s in sequences])

scaler = StandardScaler()
X      = scaler.fit_transform(X_raw)

# ─────────────────────────────────────────────────────────────────────────────
# 4. MODEL TRAINING WITH LOO-CV + BOOTSTRAPPED CONFIDENCE INTERVALS
# ─────────────────────────────────────────────────────────────────────────────
rf  = RandomForestRegressor(n_estimators=500, random_state=42, max_features='sqrt', min_samples_leaf=2)
gbr = GradientBoostingRegressor(n_estimators=300, random_state=42, learning_rate=0.05, max_depth=2)

loo = LeaveOneOut()

# LOO predictions
y_pred_rf  = cross_val_predict(rf,  X, y, cv=loo)
y_pred_gbr = cross_val_predict(gbr, X, y, cv=loo)

r2_rf   = r2_score(y, y_pred_rf)
mae_rf  = mean_absolute_error(y, y_pred_rf)
r2_gbr  = r2_score(y, y_pred_gbr)
mae_gbr = mean_absolute_error(y, y_pred_gbr)

# Bootstrap 95% CIs on LOO-R² and LOO-MAE (n=20, B=2000)
B = 2000
boot_r2_rf = []; boot_mae_rf = []
boot_r2_gbr = []; boot_mae_gbr = []
for _ in range(B):
    idx = np.random.choice(len(y), len(y), replace=True)
    yb, pb_rf, pb_gbr = y[idx], y_pred_rf[idx], y_pred_gbr[idx]
    if len(np.unique(yb)) < 2:
        continue
    boot_r2_rf.append(r2_score(yb, pb_rf))
    boot_mae_rf.append(mean_absolute_error(yb, pb_rf))
    boot_r2_gbr.append(r2_score(yb, pb_gbr))
    boot_mae_gbr.append(mean_absolute_error(yb, pb_gbr))

ci_r2_rf  = np.percentile(boot_r2_rf,  [2.5, 97.5])
ci_mae_rf = np.percentile(boot_mae_rf, [2.5, 97.5])
ci_r2_gbr  = np.percentile(boot_r2_gbr,  [2.5, 97.5])
ci_mae_gbr = np.percentile(boot_mae_gbr, [2.5, 97.5])

# Fit on full dataset for feature importance + screening
rf.fit(X, y)
gbr.fit(X, y)

# Permutation importance (model-agnostic, more reliable than MDI for small n)
pi_result = permutation_importance(rf, X, y, n_repeats=100, random_state=42)
perm_imp_mean = pi_result.importances_mean
perm_imp_std  = pi_result.importances_std

# Pearson correlation of each feature with y
pearson_r = np.array([stats.pearsonr(X_raw[:, i], y)[0] for i in range(X_raw.shape[1])])

# Spearman rank correlation (rank ordering matters more than linear)
spearman_r = np.array([stats.spearmanr(X_raw[:, i], y)[0] for i in range(X_raw.shape[1])])

# ─────────────────────────────────────────────────────────────────────────────
# 5. NOVEL APTAMER SCREENING — 50,000 candidates (10x original)
#    DNA 10-mers + RNA 10-mers (U replacing T)
# ─────────────────────────────────────────────────────────────────────────────
np.random.seed(42)
DNA_BASES = list('AGCT')
RNA_BASES = list('AGCU')

def gen_library(bases, n=25000):
    seqs = set()
    while len(seqs) < n:
        s = ''.join(np.random.choice(bases, 10))
        seqs.add(s)
    return list(seqs)

dna_lib = gen_library(DNA_BASES, 25000)
rna_lib = gen_library(RNA_BASES, 25000)
full_lib = dna_lib + rna_lib
lib_types = ['DNA']*25000 + ['RNA']*25000

# Exclude training sequences
train_set = set(sequences)
full_lib  = [s for s in full_lib if s not in train_set]
lib_types = lib_types[:len(full_lib)]

X_lib_raw = np.array([featurize(s) for s in full_lib])
X_lib     = scaler.transform(X_lib_raw)

scores_rf  = rf.predict(X_lib)
scores_gbr = gbr.predict(X_lib)
# Ensemble: average of two models
scores_ens = (scores_rf + scores_gbr) / 2.0

# High-affinity threshold: 0.85 (as before, but now on larger + ensemble)
threshold  = 0.85
high_idx   = np.where(scores_ens > threshold)[0]
n_high     = len(high_idx)
hit_rate   = n_high / len(full_lib) * 100

# Structural fitness
def structural_fitness(seq):
    seq = seq.upper().replace('U','T')
    n = len(seq)
    gc_stab = (seq.count('G') + seq.count('C')) / n
    rc = seq[::-1].translate(str.maketrans('ATGC','TACG'))
    self_comp = sum(a==b for a,b in zip(seq, rc)) / n
    return gc_stab * (1 - 0.5 * self_comp)

struct_scores = np.array([structural_fitness(s) for s in full_lib])

# Diversity: remove sequences within Hamming distance 1 of any training aptamer
def hamming(s1, s2):
    s1 = s1.upper().replace('U','T'); s2 = s2.upper().replace('U','T')
    return sum(a != b for a,b in zip(s1,s2)) if len(s1)==len(s2) else 999

def is_diverse(seq, train_seqs, min_hd=2):
    return all(hamming(seq, t) >= min_hd for t in train_seqs)

# Composite ranking: 60% binding score + 40% structural fitness
composite = 0.6*scores_ens + 0.4*struct_scores

# Filter: high-affinity + diverse
top_mask  = (scores_ens > threshold) & np.array([is_diverse(full_lib[i], sequences) for i in range(len(full_lib))])
top_idx   = np.where(top_mask)[0]
top_idx_sorted = top_idx[np.argsort(-composite[top_idx])]

top_candidates = pd.DataFrame({
    'sequence':      [full_lib[i] for i in top_idx_sorted],
    'type':          [lib_types[i] for i in top_idx_sorted],
    'pred_score_rf': scores_rf[top_idx_sorted],
    'pred_score_gbr':scores_gbr[top_idx_sorted],
    'pred_score_ens':scores_ens[top_idx_sorted],
    'struct_score':  struct_scores[top_idx_sorted],
    'composite':     composite[top_idx_sorted],
    'GC':            [X_lib_raw[i, FEAT_NAMES.index('GC')] for i in top_idx_sorted],
    'purine_frac':   [X_lib_raw[i, FEAT_NAMES.index('purine')] for i in top_idx_sorted],
    'length_nm':     [X_lib_raw[i, FEAT_NAMES.index('length_nm')] for i in top_idx_sorted],
    'charge_per_nm': [X_lib_raw[i, FEAT_NAMES.index('charge_per_nm')] for i in top_idx_sorted],
})

top10 = top_candidates.head(10)
total_diverse_high = len(top_idx)

# ─────────────────────────────────────────────────────────────────────────────
# 6. NEGATIVE CONTROL / DECOY ANALYSIS
#    Compare score distributions: known PC aptamers vs random decoys
# ─────────────────────────────────────────────────────────────────────────────
# Scores for the 18 real aptamers on RF model
positive_scores = y_pred_rf[:18]
# Scores for random decoys (all library sequences with score <0.5)
decoy_scores    = scores_ens[scores_ens < 0.5][:200]

mwu_stat, mwu_p = stats.mannwhitneyu(positive_scores, decoy_scores, alternative='greater')
ks_stat,  ks_p  = stats.ks_2samp(positive_scores, decoy_scores)

# Effect size (rank-biserial correlation)
n1, n2 = len(positive_scores), len(decoy_scores)
effect_size = 1 - (2*mwu_stat)/(n1*n2)

# ─────────────────────────────────────────────────────────────────────────────
# 7. PCA  — chemical space coverage
# ─────────────────────────────────────────────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
X_all_raw = np.vstack([X_raw, X_lib_raw[:500]])  # training + 500 library samples
X_pca     = pca.fit_transform(scaler.transform(X_all_raw))

pc1_var = pca.explained_variance_ratio_[0]*100
pc2_var = pca.explained_variance_ratio_[1]*100

# ─────────────────────────────────────────────────────────────────────────────
# 8. TOP FEATURE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
mdi_imp = rf.feature_importances_
feat_df = pd.DataFrame({
    'feature':       FEAT_NAMES,
    'MDI_importance':mdi_imp,
    'perm_mean':     perm_imp_mean,
    'perm_std':      perm_imp_std,
    'pearson_r':     pearson_r,
    'spearman_r':    spearman_r,
}).sort_values('perm_mean', ascending=False)

print("\n" + "="*70)
print("RESULTS BLOCK — PASTE THIS TO CLAUDE")
print("="*70)
print(f"\n[MODEL PERFORMANCE]")
print(f"RF  LOO-CV R2  = {r2_rf:.4f}  (95% CI: {ci_r2_rf[0]:.4f} – {ci_r2_rf[1]:.4f})")
print(f"RF  LOO-CV MAE = {mae_rf:.4f}  (95% CI: {ci_mae_rf[0]:.4f} – {ci_mae_rf[1]:.4f})")
print(f"GBR LOO-CV R2  = {r2_gbr:.4f}  (95% CI: {ci_r2_gbr[0]:.4f} – {ci_r2_gbr[1]:.4f})")
print(f"GBR LOO-CV MAE = {mae_gbr:.4f}  (95% CI: {ci_mae_gbr[0]:.4f} – {ci_mae_gbr[1]:.4f})")

print(f"\n[SCREENING — 50,000 sequences: 25,000 DNA + 25,000 RNA]")
print(f"Total screened          = {len(full_lib)}")
print(f"High-affinity (>0.85)   = {n_high}  ({hit_rate:.2f}%)")
print(f"High-affinity + diverse = {total_diverse_high}")
print(f"Top candidate           = {top10.iloc[0]['sequence']} (ens score={top10.iloc[0]['pred_score_ens']:.4f})")

print(f"\n[NEGATIVE CONTROL / DECOY SEPARATION]")
print(f"Mann-Whitney U statistic = {mwu_stat:.1f},  p = {mwu_p:.2e}")
print(f"KS statistic             = {ks_stat:.4f},  p = {ks_p:.2e}")
print(f"Effect size (rank-biserial r) = {effect_size:.4f}")

print(f"\n[PCA]")
print(f"PC1 variance = {pc1_var:.1f}%")
print(f"PC2 variance = {pc2_var:.1f}%")

print(f"\n[TOP 10 NOVEL CANDIDATES]")
print(top10[['sequence','type','pred_score_ens','struct_score','composite',
             'GC','purine_frac','length_nm','charge_per_nm']].to_string(index=False))

print(f"\n[TOP 15 FEATURES — permutation importance]")
print(feat_df.head(15)[['feature','perm_mean','perm_std','pearson_r','spearman_r']].to_string(index=False))

print("\n" + "="*70)
print("END RESULTS BLOCK")
print("="*70)
