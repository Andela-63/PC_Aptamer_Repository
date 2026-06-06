"""
screening.py
============
Virtual library generation, ensemble scoring, diversity filtering,
and composite ranking for PC-targeting aptamer candidate discovery.

Author  : AbuHuwaiz Al-Rashidi
ORCID   : 0000-0002-6615-0664
License : MIT
"""

import numpy as np
import pandas as pd
from scipy import stats


# ── Structural fitness ───────────────────────────────────────────────────────

def structural_fitness(sequence: str) -> float:
    """
    Compute structural fitness score S = GC_stability × (1 − 0.5 × self_complementarity).

    Higher score = thermodynamically stable + low self-complementarity
    (minimal hairpin formation, maximising binding-site accessibility).

    Parameters
    ----------
    sequence : str

    Returns
    -------
    float in [0, 1]
    """
    seq = sequence.upper().replace('U', 'T')
    n   = len(seq)
    gc_stab = (seq.count('G') + seq.count('C')) / n
    rc = seq[::-1].translate(str.maketrans('ATGC', 'TACG'))
    self_comp = sum(a == b for a, b in zip(seq, rc)) / n
    return gc_stab * (1 - 0.5 * self_comp)


# ── Diversity filtering ───────────────────────────────────────────────────────

def hamming_distance(seq1: str, seq2: str) -> int:
    """Hamming distance between two equal-length sequences (U→T normalised)."""
    s1 = seq1.upper().replace('U', 'T')
    s2 = seq2.upper().replace('U', 'T')
    if len(s1) != len(s2):
        return 9999
    return sum(a != b for a, b in zip(s1, s2))


def is_diverse(sequence: str,
               reference_seqs: list,
               min_hamming: int = 2) -> bool:
    """Return True if sequence is at least min_hamming from all references."""
    return all(hamming_distance(sequence, ref) >= min_hamming
               for ref in reference_seqs)


# ── Library generation ────────────────────────────────────────────────────────

def generate_library(n_dna: int = 25000,
                     n_rna: int = 25000,
                     length: int = 10,
                     random_state: int = 42) -> tuple:
    """
    Generate a stochastic virtual library of DNA and RNA aptamer sequences.

    Parameters
    ----------
    n_dna  : int   Number of DNA sequences to generate
    n_rna  : int   Number of RNA sequences to generate
    length : int   Sequence length (default 10-mer)
    random_state : int

    Returns
    -------
    sequences : list of str
    types     : list of str ('DNA' or 'RNA')
    """
    rng = np.random.default_rng(random_state)
    dna_bases = list('AGCT')
    rna_bases = list('AGCU')

    dna_seqs = set()
    while len(dna_seqs) < n_dna:
        dna_seqs.add(''.join(rng.choice(dna_bases, length)))

    rna_seqs = set()
    while len(rna_seqs) < n_rna:
        rna_seqs.add(''.join(rng.choice(rna_bases, length)))

    sequences = list(dna_seqs) + list(rna_seqs)
    types     = ['DNA'] * len(dna_seqs) + ['RNA'] * len(rna_seqs)
    return sequences, types


def screen_library(rf, gbr, scaler,
                   sequences: list,
                   types: list,
                   featurize_fn,
                   training_seqs: list,
                   binding_threshold: float = 0.85,
                   min_hamming: int = 2,
                   composite_alpha: float = 0.6) -> pd.DataFrame:
    """
    Score, filter, and rank a virtual library.

    Parameters
    ----------
    rf, gbr          : fitted sklearn estimators
    scaler           : fitted StandardScaler
    sequences        : list of str
    types            : list of str ('DNA'/'RNA')
    featurize_fn     : callable — converts sequence to feature vector
    training_seqs    : list of str — exclude/filter against these
    binding_threshold: float — minimum ensemble score for 'high-affinity' label
    min_hamming      : int — minimum Hamming distance from training sequences
    composite_alpha  : float — weight of binding score in composite rank (1−α = struct weight)

    Returns
    -------
    pd.DataFrame of all screened candidates, sorted by composite score (desc).
    """
    # Exclude training sequences
    pairs    = [(s, t) for s, t in zip(sequences, types) if s not in set(training_seqs)]
    seqs_flt = [p[0] for p in pairs]
    type_flt = [p[1] for p in pairs]

    print(f"Featurizing {len(seqs_flt):,} library sequences...")
    X_lib = scaler.transform(np.array([featurize_fn(s) for s in seqs_flt]))

    scores_rf  = rf.predict(X_lib)
    scores_gbr = gbr.predict(X_lib)
    scores_ens = (scores_rf + scores_gbr) / 2.0
    struct_sc  = np.array([structural_fitness(s) for s in seqs_flt])
    composite  = composite_alpha * scores_ens + (1 - composite_alpha) * struct_sc

    print(f"Applying threshold (>{binding_threshold}) and Hamming filter (≥{min_hamming})...")
    keep = []
    for i, seq in enumerate(seqs_flt):
        if scores_ens[i] > binding_threshold and is_diverse(seq, training_seqs, min_hamming):
            keep.append(i)

    df = pd.DataFrame({
        'sequence':      [seqs_flt[i] for i in keep],
        'type':          [type_flt[i] for i in keep],
        'score_rf':      scores_rf[keep],
        'score_gbr':     scores_gbr[keep],
        'score_ensemble':scores_ens[keep],
        'struct_score':  struct_sc[keep],
        'composite':     composite[keep],
    }).sort_values('composite', ascending=False).reset_index(drop=True)

    df.index = df.index + 1   # 1-based rank
    df.index.name = 'rank'

    print(f"\nScreening summary:")
    print(f"  Total screened      : {len(seqs_flt):,}")
    print(f"  Above threshold     : {(scores_ens > binding_threshold).sum():,}")
    print(f"  After Hamming filter: {len(df):,}")
    print(f"  Top candidate       : {df.iloc[0]['sequence']} "
          f"(ensemble = {df.iloc[0]['score_ensemble']:.4f})")
    return df


def decoy_discrimination_test(known_scores: np.ndarray,
                               decoy_scores: np.ndarray) -> dict:
    """
    Statistical discrimination test: known aptamers vs. random decoys.

    Returns a dict with Mann-Whitney U, KS test, and effect size.
    """
    mwu_stat, mwu_p = stats.mannwhitneyu(known_scores, decoy_scores,
                                          alternative='greater')
    ks_stat,  ks_p  = stats.ks_2samp(known_scores, decoy_scores)
    n1, n2 = len(known_scores), len(decoy_scores)
    effect_size = 1 - (2 * mwu_stat) / (n1 * n2)

    results = {
        'mwu_statistic': mwu_stat,
        'mwu_p':         mwu_p,
        'ks_statistic':  ks_stat,
        'ks_p':          ks_p,
        'effect_size_r': effect_size,
    }

    print("\nNEGATIVE CONTROL / DECOY DISCRIMINATION")
    print(f"  Mann-Whitney U = {mwu_stat:.1f},  p = {mwu_p:.2e}")
    print(f"  KS statistic   = {ks_stat:.4f},  p = {ks_p:.2e}")
    print(f"  Effect size r  = {effect_size:.4f}")
    return results


if __name__ == "__main__":
    print("Screening module loaded. Run notebooks/PC_Aptamer_Pipeline.ipynb for full analysis.")
