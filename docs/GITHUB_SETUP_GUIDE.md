# GitHub Repository Setup Guide
## Complete Step-by-Step Instructions for an Academically Aligned Repository

**Author:** AbuHuwaiz Al-Rashidi  
**Repository:** `PC-Aptamer-ML-Pipeline`  
**ORCID:** 0000-0002-6615-0664

---

## PART 1 — Create Your GitHub Account (if you don't have one)

1. Go to **https://github.com**
2. Click **Sign up**
3. Enter your details:
   - **Username:** `AbuHuwaizAlRashidi`  ← use exactly this (it matches the URL cited in your paper)
   - **Email:** a.alrashidi@ksu.edu.sa
   - **Password:** choose a strong password
4. Verify your email address
5. Select the **Free** plan

---

## PART 2 — Set Up Your GitHub Profile (Academic Standard)

1. Click your profile picture (top right) → **Settings**
2. Fill in:
   - **Name:** AbuHuwaiz Al-Rashidi
   - **Bio:** Computational biologist | King Saud University | Aptamer design | ML in drug discovery
   - **Company:** King Saud University
   - **Location:** Riyadh, Saudi Arabia
   - **Website:** (your institution page if available)
3. Click **Save**

---

## PART 3 — Create the Repository

1. Click the **+** button (top right) → **New repository**

2. Fill in the form exactly as follows:

   | Field | Value |
   |-------|-------|
   | **Repository name** | `PC-Aptamer-ML-Pipeline` |
   | **Description** | ML-enhanced SCIA pipeline for phosphatidylcholine-targeting aptamer design — open-source Python + Google Colab |
   | **Visibility** | ✅ Public |
   | **Add a README file** | ☐ Leave UNCHECKED (we will upload our own) |
   | **Add .gitignore** | ☐ Leave UNCHECKED |
   | **Choose a license** | ☐ Leave UNCHECKED (our LICENSE file is included) |

3. Click **Create repository**

4. You will see an empty repository page. **Leave this tab open.**

---

## PART 4 — Upload All Files (No Git Knowledge Required)

GitHub allows direct file upload from your browser. Follow this order carefully.

### Step 4a — Upload the README first

1. On your empty repository page, click **uploading an existing file** (blue link in the middle of the page)
2. Drag and drop **`README.md`** from your downloaded files
3. At the bottom, under *Commit changes*:
   - Title: `Add README`
   - Leave description blank
4. Click **Commit changes**

---

### Step 4b — Create the folder structure and upload all files

GitHub creates folders automatically when you include them in the file path.
Follow these steps for each folder:

#### Upload `notebooks/` folder

1. Click **Add file** → **Upload files**
2. Drag and drop: **`PC_Aptamer_Pipeline.ipynb`**
3. Before committing, click the file name in the upload area and **rename it** to:
   ```
   notebooks/PC_Aptamer_Pipeline.ipynb
   ```
   (Type `notebooks/` before the filename — GitHub will create the folder)
4. Commit message: `Add Colab notebook`
5. Click **Commit changes**

#### Upload `data/` folder

1. Click **Add file** → **Upload files**
2. Drag both files: `training_aptamers.csv` and `top10_candidates.csv`
3. Rename each to `data/training_aptamers.csv` and `data/top10_candidates.csv`
4. Commit message: `Add training and candidate data`
5. Click **Commit changes**

#### Upload `scripts/` folder

1. Click **Add file** → **Upload files**
2. Drag all five script files:
   - `scia_energy.py`
   - `feature_engineering.py`
   - `ml_ensemble.py`
   - `screening.py`
   - `generate_figures.py`
3. Rename each with prefix `scripts/` (e.g. `scripts/scia_energy.py`)
4. Commit message: `Add Python pipeline scripts`
5. Click **Commit changes**

#### Upload `figures/` folder

1. Click **Add file** → **Upload files**
2. Drag all five PNG files:
   - `Figure1_SCI_Profiles.png`
   - `Figure2_Model_Performance.png`
   - `Figure3_Feature_Importance.png`
   - `Figure4_Screening.png`
   - `Figure5_PCA_Decoy.png`
3. Rename each with prefix `figures/`
4. Commit message: `Add publication figures`
5. Click **Commit changes**

#### Upload `docs/` folder

1. Click **Add file** → **Upload files**
2. Drag: `experimental_roadmap.md`
3. Rename to `docs/experimental_roadmap.md`
4. Commit message: `Add experimental validation roadmap`
5. Click **Commit changes**

#### Upload root-level files

1. Click **Add file** → **Upload files**
2. Drag: `requirements.txt` and `LICENSE`
3. Do NOT add any folder prefix — these go in the root
4. Commit message: `Add requirements and license`
5. Click **Commit changes**

---

## PART 5 — Add the "Open in Colab" Button (Verify it Works)

1. After uploading the notebook, go to:
   ```
   https://colab.research.google.com/github/AbuHuwaizAlRashidi/PC-Aptamer-ML-Pipeline/blob/main/notebooks/PC_Aptamer_Pipeline.ipynb
   ```
2. The notebook should open in Colab automatically
3. Click **Runtime → Run all** to verify it runs without errors
4. If it works, the Colab badge in your README is live ✓

---

## PART 6 — Configure Repository Settings (Academic Standard)

### Add Topics (keywords for discoverability)

1. On your repository main page, click the ⚙️ gear icon next to **About** (top right)
2. Under **Topics**, add these keywords one by one:
   ```
   aptamer-design
   phosphatidylcholine
   screened-coulomb-interaction
   machine-learning
   random-forest
   gradient-boosting
   bioinformatics
   drug-design
   open-source
   google-colab
   python
   computational-biology
   ```
3. Tick **Releases**, **Packages**, **Environments** if available
4. Click **Save changes**

### Add a Website URL

In the same **About** panel, paste the Colab URL as the website:
```
https://colab.research.google.com/github/AbuHuwaizAlRashidi/PC-Aptamer-ML-Pipeline/blob/main/notebooks/PC_Aptamer_Pipeline.ipynb
```

---

## PART 7 — Create a Release (Required for Zenodo DOI)

A release creates a permanent, citable snapshot of your code.

1. On your repository page, click **Releases** (right sidebar) → **Create a new release**
2. Fill in:
   - **Tag version:** `v1.0.0`
   - **Release title:** `v1.0.0 — Initial release accompanying submitted manuscript`
   - **Description:**
     ```
     First public release of the ML-Enhanced SCIA pipeline for PC-targeting aptamer design.

     Includes:
     - Complete Google Colab notebook (16 cells, fully executable)
     - All 5 Python pipeline modules
     - Training dataset (18 aptamers + 2 controls)
     - Top 10 novel candidate sequences
     - 5 publication-quality figures (300 DPI)
     - Experimental validation roadmap

     Associated manuscript: Al-Rashidi A. (2026). ML-Enhanced SCIA for 
     PC-Targeting Aptamer Design. Submitted.
     ORCID: 0000-0002-6615-0664
     ```
3. Click **Publish release**

---

## PART 8 — Get a Zenodo DOI (Permanent Academic Citation)

Zenodo is CERN's open-science archive — free, permanent, and accepted by all journals.

1. Go to **https://zenodo.org**
2. Click **Log in** → **Log in with GitHub**
3. Authorise Zenodo to access your GitHub account
4. Go to: **https://zenodo.org/account/settings/github/**
5. Find `PC-Aptamer-ML-Pipeline` in the list → Toggle it **ON**
6. Go back to GitHub and click your **v1.0.0 release** — Zenodo will automatically detect it
7. Go back to Zenodo → **Upload** → find your repository → click **Publish**
8. Zenodo assigns a DOI in the format: `10.5281/zenodo.XXXXXXX`
9. **Copy this DOI** and update it in:
   - `README.md` (replace `10.5281/zenodo.XXXXXXX`)
   - Your manuscript (Data Availability Statement and Supplementary Materials section)

---

## PART 9 — Final Checklist Before Submission

Run through this checklist before submitting your manuscript:

| # | Item | Status |
|---|------|--------|
| 1 | Repository is **Public** | ☐ |
| 2 | README displays correctly with all badges | ☐ |
| 3 | Colab badge opens notebook in one click | ☐ |
| 4 | Notebook runs end-to-end without errors | ☐ |
| 5 | All 5 figures are in `figures/` folder | ☐ |
| 6 | `data/training_aptamers.csv` is present | ☐ |
| 7 | `data/top10_candidates.csv` is present | ☐ |
| 8 | `requirements.txt` lists all dependencies | ☐ |
| 9 | `LICENSE` (MIT) is present | ☐ |
| 10 | Repository has ≥ 10 topic tags | ☐ |
| 11 | v1.0.0 Release is published | ☐ |
| 12 | Zenodo DOI is obtained and updated in manuscript | ☐ |
| 13 | Zenodo DOI updated in `README.md` | ☐ |
| 14 | ORCID `0000-0002-6615-0664` appears in README | ☐ |
| 15 | Reference [10] DOI `10.1016/j.medidd.2026.100247` is correct | ☐ |

---

## Your Final Repository URL

```
https://github.com/AbuHuwaizAlRashidi/PC-Aptamer-ML-Pipeline
```

This is the URL cited in your manuscript. Make sure it matches exactly.

---

## Support

If you encounter any issues:
- GitHub documentation: https://docs.github.com
- Zenodo documentation: https://help.zenodo.org
- Colab documentation: https://colab.research.google.com/notebooks/intro.ipynb
