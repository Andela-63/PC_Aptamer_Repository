# Experimental Validation Roadmap

**Pipeline:** ML-Enhanced SCIA for PC-Targeting Aptamers  
**Author:** AbuHuwaiz Al-Rashidi | ORCID: 0000-0002-6615-0664  
**Reference:** Manuscript submitted June 2026

---

## Overview

This document specifies a five-assay experimental protocol for validating the
top 10 computationally predicted PC-targeting aptamer candidates listed in
`data/top10_candidates.csv`. The protocol is designed so that computational
predictions can be tested and falsified in a standard molecular biology
wet-lab setting.

**Priority candidates for immediate synthesis:**

| Rank | Sequence | Type | Composite Score |
|------|----------|------|----------------|
| 1 | GAAGCAGGCG | DNA | 0.794 |
| 2 | GGCCAGAAAG | DNA | 0.787 |
| 3 | AAAGCCGGAG | RNA | 0.785 |
| 4 | AACGAGGAGC | DNA | 0.782 |
| 5 | GGCCAGAAGA | DNA | 0.782 |

---

## Assay 1 — Surface Plasmon Resonance (SPR)

**Instrument:** Biacore T200 or equivalent  
**Purpose:** Measure real Kd values; primary falsification of computational scores

### Protocol

1. **Liposome preparation:** Extrude DPPC/cholesterol (70:30 mol%) in PBS
   (pH 7.4) through 100 nm membranes. Verify size by DLS (target: 100 ± 20 nm).
2. **Chip preparation:** Capture liposomes on L1 sensor chip
   (flow rate 5 µL/min; 2,000 RU target).
3. **Aptamer synthesis:** Order top 10 candidates from Integrated DNA
   Technologies (IDT) or equivalent at 25 nmol scale, standard desalting.
   RNA candidates: order with 2'-OH intact.
4. **Binding assay:** Six-point concentration series (1, 5, 25, 100, 250,
   500 nM) in running buffer (PBS + 0.05% Tween-20). Contact time: 120 s;
   dissociation: 180 s. Regenerate with 0.5% SDS (30 s).
5. **Analysis:** Fit 1:1 Langmuir model. Report ka, kd, Kd ± SE.

**Acceptance criterion:** Kd < 500 nM constitutes confirmed high-affinity binding.

---

## Assay 2 — Fluorescence Polarisation Anisotropy (FPA)

**Purpose:** Orthogonal confirmation of SPR Kd values; higher throughput

### Protocol

1. Synthesise top 5 candidates with 5'-FITC label (IDT or equivalent).
2. Prepare 8-point titration series of DPPC liposomes (1 nM–1 µM) in black
   96-well plates.
3. Incubate labelled aptamer (10 nM) with each liposome concentration (30 min,
   37 °C, protected from light).
4. Measure anisotropy on plate reader: λex = 490 nm, λem = 535 nm.
5. Fit binding curve to one-site saturation equation to derive apparent Kd.

---

## Assay 3 — Cell-Surface Binding (Flow Cytometry)

**Purpose:** Confirm binding to PC-rich cancer cell surfaces; test
cell-selectivity relative to normal epithelium

### Cell lines

| Line | Description | Expected PC expression |
|------|-------------|----------------------|
| MCF-7 | Breast cancer | High (PC-enriched outer leaflet) |
| MCF-10A | Normal mammary epithelium | Lower |

### Protocol

1. Grow cells to 70% confluency; detach with EDTA (not trypsin — protects
   surface lipids).
2. Wash with PBS; resuspend at 1×10⁶ cells/mL in staining buffer
   (PBS + 2% FBS).
3. Incubate with FITC-labelled aptamer at 100 nM (30 min, 4 °C, rotating).
4. Wash 3× with PBS; fix with 2% PFA.
5. Analyse on flow cytometer (FITC channel); gate on live singlets.
6. Compute MCF-7/MCF-10A median fluorescence ratio as PC-specificity index.
   Target: ratio > 3 for positive hit.

---

## Assay 4 — Circular Dichroism (CD) Spectroscopy

**Purpose:** Confirm that top candidates adopt a folded tertiary structure
rather than unstructured coil (required for reliable binding)

### Protocol

1. Dissolve top 3 aptamers at 5 µM in 10 mM sodium phosphate buffer (pH 7.0)
   + 100 mM KCl.
2. Heat to 95 °C for 5 min; cool slowly to 25 °C (annealing).
3. Record CD spectra (190–300 nm; 1 mm path length cuvette).
4. Interpret: Positive peak at 270 nm + negative at 240 nm = parallel
   G-quadruplex character (favourable for PC binding); random coil signature
   (large negative peak at 200 nm) = unfavourable.

---

## Assay 5 — Lipid Selectivity Competition Panel

**Purpose:** Confirm specificity for PC head group over other major
membrane phospholipids

### Lipids to test

| Lipid | Abbreviation | Head group |
|-------|-------------|------------|
| Phosphatidylethanolamine | PE | Ethanolamine |
| Phosphatidylserine | PS | Serine (anionic) |
| Sphingomyelin | SM | Phosphocholine (similar to PC) |
| Phosphatidylcholine | PC | Choline (target) |

### Protocol

1. Prepare unilamellar vesicles for each lipid type (single-component, 100 nm).
2. Run SPR competition experiment: inject aptamer (100 nM) first onto
   PC chip; then repeat with PE, PS, SM chips in same conditions.
3. Report relative response (RU on each chip).
4. Acceptance criterion: PC response ≥ 3× any competitor lipid.

---

## Data Reporting Standards

All experimental results should be reported as:
- Mean ± SD from at least **n = 3 independent experiments**
- Kd values with **95% confidence intervals** from curve fitting
- Statistical comparisons using **unpaired t-test or Mann–Whitney U** as appropriate
- Raw data deposited in **Figshare or Zenodo** alongside the computational data

---

## Contact

For experimental collaboration inquiries:  
**AbuHuwaiz Al-Rashidi** | a.alrashidi@ksu.edu.sa  
King Saud University, Riyadh, Saudi Arabia
