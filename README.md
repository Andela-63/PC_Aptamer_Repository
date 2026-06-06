# ML-Enhanced Screened Coulomb Interaction Approach for PC-Targeting Aptamer Design

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Colab](https://img.shields.io/badge/Open%20in-Colab-orange?logo=googlecolab)](https://colab.research.google.com/github/Kolawole_Sherifdeen_kehinde/PC-Aptamer-ML-Pipeline/blob/main/notebooks/PC_Aptamer_Pipeline.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.XXXXXXX-blue)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![ORCID](https://img.shields.io/badge/ORCID-0000--0002--6615--0664-brightgreen?logo=orcid)](https://orcid.org/0000-0002-6615-0664)

---

## Overview

This repository contains the **complete open-source Python pipeline** accompanying the manuscript:

> **Kolawole Sherifdeen Kehinde.** *Machine Learning-Enhanced Screened Coulomb Interaction Approach for Predictive Design and Optimization of Phosphatidylcholine-Targeting Aptamers: A Fully Open-Source Computational Pipeline with Experimental Validation Roadmap.* (2026). Submitted.

### What this pipeline does

| Step | Module | Description |
|------|--------|-------------|
| 1 | SCI Energy Calculator | Reimplements Alharbi et al. [2026] SCIA physics in Python — no Mathematica required |
| 2 | Feature Engineering | Encodes aptamers as 31-dimensional physicochemical vectors including SCI energy terms |
| 3 | ML Ensemble | RF + Gradient Boosting with LOO-CV and bootstrapped 95% confidence intervals |
| 4 | Screening | 50,000 novel 10-mer DNA and RNA candidates scored and ranked |
| 5 | Diversity Filtering | Hamming-distance filter ensures genuine chemical novelty |
| 6 | Visualisation | All 5 publication-quality figures generated automatically |

### Key results

- **GBR LOO-CV R² = 0.473** (95% CI: 0.281–0.831); MAE = 0.084 (95% CI: 0.027–0.160)
- **6,744 high-affinity, diverse candidates** identified from 50,000 screened sequences
- **Top candidate: GAAGCAGGCG** (DNA; ensemble score = 0.857; composite = 0.794)
- **Perfect decoy discrimination**: Mann–Whitney U p = 1.10×10⁻¹², effect size r = 1.00
- First pipeline to screen **RNA aptamers** against PC using SCIA-derived features

---

## Repository Structure

```
PC-Aptamer-ML-Pipeline/
│
├── notebooks/
│   └── PC_Aptamer_Pipeline.ipynb      # Main Colab notebook (full pipeline)
│
├── data/
│   ├── training_aptamers.csv          # 18 PC aptamers + 2 negative controls
│   └── top10_candidates.csv           # Top 10 novel predicted candidates
│
├── figures/
│   ├── Figure1_SCI_Profiles.png       # SCI energy profiles
│   ├── Figure2_Model_Performance.png  # LOO-CV scatter + bootstrapped CIs
│   ├── Figure3_Feature_Importance.png # Three-method feature importance
│   ├── Figure4_Screening.png          # 50,000-candidate screening results
│   └── Figure5_PCA_Decoy.png          # PCA + decoy discrimination
│
├── scripts/
│   ├── scia_energy.py                 # SCI energy calculator module
│   ├── feature_engineering.py         # 31-D feature vector builder
│   ├── ml_ensemble.py                 # RF + GBR ensemble with bootstrapped CV
│   ├── screening.py                   # Virtual library screening
│   └── generate_figures.py            # Publication figure generator
│
├── docs/
│   └── experimental_roadmap.md        # Five-assay experimental validation protocol
│
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
└── README.md                          # This file
```

---

## Quick Start

### Option 1 — Google Colab (recommended, zero installation)

Click the badge:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github//PC-Aptamer-ML-Pipeline/blob/main/notebooks/PC_Aptamer_Pipeline.ipynb)

All dependencies install automatically in the first cell.

### Option 2 — Local installation

```bash
# 1. Clone the repository
git clone https://github.com/Kolawole Sherifdeen Kehinde/PC-Aptamer-ML-Pipeline.git
cd PC-Aptamer-ML-Pipeline

# 2. Create a virtual environment (recommended)
python3 -m venv aptamer_env
source aptamer_env/bin/activate          # Linux / macOS
aptamer_env\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline
python scripts/scia_energy.py
python scripts/feature_engineering.py
python scripts/ml_ensemble.py
python scripts/screening.py
python scripts/generate_figures.py
```

---

## Data

### Training dataset (`data/training_aptamers.csv`)

18 PC-targeting aptamers and 2 negative controls from Alharbi et al. [2026] [doi:10.1016/j.medidd.2026.100247], with normalised binding scores.

| Column | Description |
|--------|-------------|
| `aptamer_id` | Label (PCAPt1–PCAPt18, NEG1–NEG2) |
| `sequence` | 10-mer nucleotide sequence |
| `binding_score` | Normalised SCIA binding score [0–1] |
| `category` | Universal / DNA / RNA / Hybrid / Negative |

### Output candidates (`data/top10_candidates.csv`)

Top 10 novel candidates from the ML screening, ready for experimental follow-up.

---

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@article{KolawoleSherifdeen2026,
  author    = {Sherifdee Kehinde, Kolawole},
  title     = {Machine Learning-Enhanced Scereened Coulomb Interaction Approach
               for Predictive Design and Optimization of Phosphatidylcholine-Targeting
               Aptamers: A Fully Open-Source Computational Pipeline with
               Experimental Validation Roadmap},
  journal   = {[Journal name upon acceptance]},
  year      = {2026},
  doi       = {10.5281/zenodo},
  orcid     = {0000-0002-6615-0664}
}
```

Please also cite the foundational SCIA paper:

```bibtex
@article{alharbi2026,
  author  = {Alharbi, Khalid K. and Alnefaie, Rawan and Almars, Amani and Abdulwahab, Tariq},
  title   = {Energy-based method for designing aptamers to target phosphatidylcholines},
  journal = {Medicine in Drug Discovery},
  year    = {2026},
  doi     = {10.1016/j.medidd.2026.100247}
}
```

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.  
You are free to use, modify, and distribute this code with attribution.

---

## Author

**Kolawole Sherifdeen Kehinde**  
Department of Computational Biology and Drug Design  
King Saud University, Riyadh, Saudi Arabia  
📧 Kolawolesherifdeen63@gmail.com  
🔗 ORCID: [0000-0002-6615-0664](https://orcid.org/0000-0002-6615-0664)

---

## Acknowledgements

- Alharbi et al. [2026] for openly reporting aptamer sequences and binding data
- The Python scientific ecosystem: NumPy, SciPy, scikit-learn, Matplotlib, Seaborn, Pandas
