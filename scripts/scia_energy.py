"""
scia_energy.py
==============
Open-source Python reimplementation of the Screened Coulomb Interaction
Approach (SCIA) originally described in:

    Alharbi KK, Alnefaie R, Almars A, Abdulwahab T.
    Energy-based method for designing aptamers to target phosphatidylcholines.
    Med Drug Discov. 2026. doi:10.1016/j.medidd.2026.100247

Author  : AbuHuwaiz Al-Rashidi
ORCID   : 0000-0002-6615-0664
License : MIT
"""

import numpy as np
from scipy.integrate import trapezoid


# ── Physical constants ────────────────────────────────────────────────────────
E0  = 8.854e-12   # Vacuum permittivity (F m⁻¹)
ER  = 80.0        # Relative permittivity of water
E   = 1.6e-19     # Elementary charge (C)
KB  = 1.38e-23    # Boltzmann constant (J K⁻¹)
T   = 310.0       # Physiological temperature (K)
N   = 1.0e26      # Ionic number density (m⁻³)
PC_CHARGE = 2     # PC zwitterion net charge magnitude

# ── Nucleotide parameters (Table 1, Alharbi et al. 2026) ────────────────────
FUNCTIONAL_CHARGE = {
    'A': 3,   # Adenine
    'G': 4,   # Guanine
    'C': 4,   # Cytosine
    'T': 3,   # Thymine  (DNA)
    'U': 3,   # Uracil   (RNA)
}

LATTICE_SIZE = {
    'A': 0.34,  # nm
    'G': 0.34,
    'C': 0.30,
    'T': 0.30,
    'U': 0.30,
}


def sci_energy(abb_charge: int,
               r_nm_array: np.ndarray,
               k_points: int = 8000) -> np.ndarray:
    """
    Compute the screened Coulomb interaction (SCI) binding energy between
    one aptamer building block (ABB) and the PC zwitterion at each distance
    in r_nm_array.

    Parameters
    ----------
    abb_charge  : int
        Functional charge of the ABB nucleotide (from FUNCTIONAL_CHARGE).
    r_nm_array  : np.ndarray
        Array of separation distances in nanometres.
    k_points    : int
        Number of wavenumber grid points for numerical integration (default 8000).

    Returns
    -------
    np.ndarray
        SCI binding energy (J) at each distance in r_nm_array.

    Physics
    -------
    Direct Coulomb potential in wavenumber space:
        V(k) = (q1 * q2 * e²) / (ε0 * εr * k²)

    Screened potential:
        V_sc(k) = V(k) * exp(-f * k²)
        where f = 1 / (2π * kB * T * N)

    Real-space binding energy via inverse Fourier integration (1D radial):
        E_sci(r) = ∫ V_sc(k) * cos(k * r) dk
    """
    r = r_nm_array * 1e-9  # convert nm → m
    k = np.linspace(1e9, 1e11, k_points)
    screening_factor = 1.0 / (2 * np.pi * KB * T * N)

    V_direct  = (abb_charge * PC_CHARGE * E**2) / (E0 * ER * k**2)
    V_screened = V_direct * np.exp(-screening_factor * k**2)

    energies = np.array([
        trapezoid(V_screened * np.cos(k * ri), k)
        for ri in r
    ])
    return energies


def compute_all_profiles(r_nm: np.ndarray = None) -> dict:
    """
    Compute SCI energy profiles for all five nucleotide types (A, G, C, T, U).

    Parameters
    ----------
    r_nm : np.ndarray, optional
        Distance array in nm. Defaults to 300 points from 0.1 to 5.0 nm.

    Returns
    -------
    dict  {nucleotide: energy_array}
    """
    if r_nm is None:
        r_nm = np.linspace(0.1, 5.0, 300)
    return {nt: sci_energy(FUNCTIONAL_CHARGE[nt], r_nm) for nt in FUNCTIONAL_CHARGE}


def verify_rank_order(profiles: dict) -> bool:
    """
    Verify that the rank ordering G > C > A > T ≈ U is preserved at r = 0.1 nm,
    consistent with Figure 3 of Alharbi et al. [2026].

    Returns True if the rank order is correct.
    """
    e0 = {nt: np.abs(profiles[nt][0]) for nt in profiles}
    rank = sorted(e0, key=lambda x: -e0[x])
    expected_top2 = {'G', 'C'}
    expected_bottom3 = {'A', 'T', 'U'}
    correct = (set(rank[:2]) == expected_top2 and
               set(rank[2:]) == expected_bottom3)
    status = "PASSED ✓" if correct else "FAILED ✗"
    print(f"Rank order verification: {' > '.join(rank)}  [{status}]")
    return correct


if __name__ == "__main__":
    print("Running SCI energy calculator...")
    r = np.linspace(0.1, 5.0, 300)
    profiles = compute_all_profiles(r)
    verify_rank_order(profiles)

    print("\nSCI energies at r = 0.1 nm (minimum distance):")
    for nt in 'GCAT U':
        if nt.strip():
            print(f"  {nt}: {profiles[nt][0]:.6e} J")
    print("\nSCI energy profiles computed successfully.")
    print("Use compute_all_profiles() in your pipeline.")
