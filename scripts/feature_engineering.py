"""
feature_engineering.py
======================
Converts raw aptamer sequences into 31-dimensional physicochemical
feature vectors for ML modelling.

Feature dimensions
------------------
 0– 4  : Mononucleotide fractions (fA, fG, fC, fT, fU)
 5     : GC content
 6     : Purine fraction (A + G)
 7     : Total functional charge
 8     : Charge density (charge / base)
 9     : Charge per nm
10     : Physical length (nm)
11     : Shannon entropy
12     : Sequence length
13–28  : 16 dinucleotide frequencies (AA, AG, ..., TT)
29–31  : SCI energy features (minimum-distance energy for A, G, C, T with PC)

Author  : AbuHuwaiz Al-Rashidi
ORCID   : 0000-0002-6615-0664
License : MIT
"""

import numpy as np
import pandas as pd
from math import log2
from scia_energy import FUNCTIONAL_CHARGE, LATTICE_SIZE, sci_energy


# ── Constants ─────────────────────────────────────────────────────────────────
DINUCLEOTIDES = [a + b for a in 'AGCT' for b in 'AGCT']  # 16 dinucs (DNA-normalised)

# Pre-compute SCI energies at minimum distance (0.1 nm) for use as features
_R_MIN = np.array([0.1])
SCI_FEATURES = {nt: sci_energy(FUNCTIONAL_CHARGE[nt], _R_MIN)[0]
                for nt in 'AGCT'}

FEATURE_NAMES = (
    ['fA', 'fG', 'fC', 'fT', 'fU',
     'GC', 'purine', 'charge_total', 'charge_dens', 'charge_per_nm',
     'length_nm', 'entropy', 'seq_len']
    + DINUCLEOTIDES
    + ['SCI_A', 'SCI_G', 'SCI_C', 'SCI_T']
)

assert len(FEATURE_NAMES) == 31, "Feature vector must be 31-dimensional"


def featurize(sequence: str) -> list:
    """
    Convert a single aptamer sequence into a 31-dimensional feature vector.

    Parameters
    ----------
    sequence : str
        Aptamer sequence (case-insensitive; RNA or DNA; U is handled natively).

    Returns
    -------
    list of float — length 31
    """
    seq = sequence.upper()
    n = len(seq)

    # Mononucleotide counts
    counts = {nt: seq.count(nt) for nt in 'AGCTU'}
    fracs  = {nt: counts[nt] / n for nt in 'AGCTU'}

    gc         = (counts['G'] + counts['C']) / n
    purine     = (counts['A'] + counts['G']) / n
    charge_tot = sum(FUNCTIONAL_CHARGE.get(nt, 3) for nt in seq)
    length_nm  = sum(LATTICE_SIZE.get(nt, 0.32) for nt in seq)
    charge_den = charge_tot / n
    charge_per_nm = charge_tot / length_nm if length_nm > 0 else 0.0

    # Shannon entropy
    entropy = -sum(
        (counts[nt] / n) * log2(counts[nt] / n)
        for nt in 'AGCTU' if counts[nt] > 0
    )

    # Dinucleotide frequencies (RNA normalised → U→T for consistent comparison)
    seq_dna = seq.replace('U', 'T')
    di_counts = {d: 0 for d in DINUCLEOTIDES}
    for i in range(len(seq_dna) - 1):
        di = seq_dna[i:i+2]
        if di in di_counts:
            di_counts[di] += 1
    total_di = max(1, len(seq_dna) - 1)
    di_vec = [di_counts[d] / total_di for d in DINUCLEOTIDES]

    # SCI energy features (4 values: A, G, C, T interactions with PC)
    sci_feats = [SCI_FEATURES[nt] for nt in 'AGCT']

    return (
        [fracs['A'], fracs['G'], fracs['C'], fracs['T'], fracs['U'],
         gc, purine, charge_tot, charge_den, charge_per_nm,
         length_nm, entropy, n]
        + di_vec
        + sci_feats
    )


def featurize_dataset(sequences: list) -> np.ndarray:
    """
    Featurize a list of sequences into a design matrix.

    Parameters
    ----------
    sequences : list of str

    Returns
    -------
    np.ndarray of shape (n_sequences, 31)
    """
    return np.array([featurize(s) for s in sequences])


def load_training_data(csv_path: str = '../data/training_aptamers.csv'):
    """
    Load and featurize the training dataset.

    Returns
    -------
    X_raw : np.ndarray (20, 31)   — raw feature matrix
    y     : np.ndarray (20,)      — binding scores
    meta  : pd.DataFrame          — labels, sequences, categories
    """
    df = pd.read_csv(csv_path)
    X_raw = featurize_dataset(df['sequence'].tolist())
    y     = df['binding_score'].values
    return X_raw, y, df


if __name__ == "__main__":
    print("Feature engineering module — self-test")
    test_seq = "GAAAAGGAGC"
    feat = featurize(test_seq)
    print(f"Sequence : {test_seq}")
    print(f"Features : {len(feat)}-dimensional")
    for name, val in zip(FEATURE_NAMES, feat):
        print(f"  {name:>15s} = {val:.6f}")
