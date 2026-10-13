# HYPERAKTIV-ML

**Beyond Binary ADHD Classification: Multimodal Phenotyping in a Clinical Psychiatric Cohort**

Siyu Li, E. Margaret Perkoff, Lyle Ungar (University of Pennsylvania)

Code for the paper published at **[MultiPsyche, 2026]** ([paper link](https://openreview.net/forum?id=8gvzO3G5Hm&noteId=8gvzO3G5Hm)).

The pipeline evaluates four measurement modalities on the public [HYPERAKTIV dataset](https://datasets.simula.no/hyperaktiv/) (Hicks et al., 2021), which contains 51 adults with ADHD and 52 psychiatric (not healthy) controls:

- **CPT-II** (Conners' Continuous Performance Test II)
- **Self-report and rating scales** (ASRS, WURS, MADRS, HADS-A, HADS-D)
- **Heart-rate variability (HRV)**
- **Wrist actigraphy**

It covers three analyses: binary ADHD vs. non-ADHD classification, three-way classification (non-inattentive ADHD / inattentive ADHD "ADD" / non-ADHD clinical controls), and unsupervised clustering.

## Main results

All values below are taken from the paper. Please read them as exploratory (small cohort, see [Limitations](#limitations)).

| Analysis | Result |
|---|---|
| Binary classification, best block (CPT-II + self-report, Logistic Regression, stratified 10-fold CV, n = 97) | AUC 0.908, balanced accuracy 0.795, F1 0.789, MCC 0.607, permutation p = 0.002. This estimate is not nested. |
| Binary classification, common complete-case sample (n = 64, nested 5×5 repeated CV) | CPT-II + self-report 0.807, self-report alone 0.799, CPT-II alone 0.784. The difference between CPT-II + self-report and self-report alone is 0.009 (95% CI [−0.152, 0.262]), so the combination is not reliably better. |
| Physiological modalities alone | HRV AUC 0.429 (at or below chance, p = 0.798). Actigraphy AUC 0.594 (not distinguishable from chance, p = 0.146). |
| Three-way classification | Macro-F1 0.630 (all modalities, n = 64) and 0.628 (self-report, n = 103). The inattentive (ADD) group has the lowest recall in every block (0.524 and 0.478 for those two blocks). Classifier choice per block is not nested. |
| Clustering (K-Means, k restricted to 3–6 by silhouette) | Clusters capture feature-space structure that only partially aligns with diagnoses (e.g., CPT-II: k = 4, ARI 0.050; HRV: highest silhouette 0.309 but ARI −0.008). These are feature-space patterns, not validated subtypes. |

## Repository structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
└── notebooks/
    └── HYPERAKTIV_ML_pipeline.ipynb   # canonical notebook (Phases 0-7)
```

## Pipeline overview

| Phase | Description | Paper |
|---|---|---|
| 0 | Install dependencies, fix random seeds (`SEED = 42`) | |
| 1 | Load and clean CPT-II, actigraphy, HRV, self-report, and patient-info tables; build per-modality participant sets | Tables 1, 7 |
| 2 | Binary classification across 9 modality combinations (LR / RF / XGBoost / LightGBM, 10-fold stratified CV, majority baselines, 500-iteration permutation tests) | Tables 2, 3, 11 |
| 2b | Common complete-case robustness analysis (n = 64, 5×5 repeated nested CV, paired AUC differences) | Tables 4, 8, 9, 10 — [notebook section / script] |
| 3 | Three-way classification with per-class analysis | Tables 5, 12, 13 |
| 4 | Per-modality K-Means clustering, k selection, ARI and chi-square against clinical labels | Tables 6, 15–18, Fig. 1 |
| 5–7 | Figures, cross-modality agreement, optional PDF export | |

CPT-II feature-level tests (Table 14) are in [notebook section / script].

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # or conda/uv
pip install -r requirements.txt
```

Last end-to-end run: [DATE], Python [VERSION].

### Data

The data are **not** included in this repository. Download HYPERAKTIV from the official sources:

- Simula: <https://datasets.simula.no/hyperaktiv/>
- OSF: <https://osf.io/3agwr>

Unzip `hyperaktiv.zip` and set `DATA_PATH` in Phase 1 of the notebook to the local folder. Please record the checksum of the archive you used (`md5sum hyperaktiv.zip`) when reporting results.

Alternatively, the notebook can download a third-party Kaggle mirror ([`arashnic/adhd-diagnosis-data`](https://www.kaggle.com/datasets/arashnic/adhd-diagnosis-data)) via `kagglehub`. This needs a free Kaggle account and API token (`~/.kaggle/kaggle.json`, or `KAGGLE_USERNAME` / `KAGGLE_KEY`). The mirror may differ from the official release, so the official source is recommended.

Please follow the dataset's terms of use on the Simula/OSF pages. Per Simula, use for competitions and commercial purposes requires prior written permission from the dataset authors.

## Reproducibility notes

- Seeds are fixed globally (`random`, `numpy`, `PYTHONHASHSEED`; `SEED = 42`).
- For supervised classification, all preprocessing that estimates parameters from data (including median imputation of missing values) is performed within training folds.
- Permutation p-values use 500 permutations of balanced accuracy, so the smallest attainable value is p = 0.002 (reported as `p = 0.002`, not `p < 0.001`).
- Raw CPT-II scores are used for classification to avoid the normative-reference information embedded in T-scores. T-scores are used for clustering so that the heterogeneous dimensions are on comparable scales for Euclidean K-Means.
- Clustering is run on all available data for each modality space (no train/test split).
- The common-sample analysis (n = 64) selects among LR, RF, XGBoost and LightGBM in an inner loop; the main binary analysis uses Logistic Regression as the reference classifier, chosen after comparing the four classifiers (Table 2), so its estimates are optimistic.

## Limitations

- The cohort is small (n = 103; 64 participants have data for all modalities), so all estimates, including AUC = 0.908, are exploratory.
- Classifier selection in the three-way analysis is not nested.
- The control group has psychiatric diagnoses, so HRV and actigraphy results may reflect shared features of other conditions rather than an absence of ADHD-related signal.
- Cluster assignments are statistical structure, not clinical subtypes.

## Ethics and intended use

This is a secondary analysis of a public, de-identified dataset; no new human-subjects data were collected. This code is research software and **must not be used as a standalone diagnostic tool**.

## Citation

If you use this code, please cite the paper and the dataset:

```bibtex
@inproceedings{li2026beyond,
  title     = {Beyond Binary {ADHD} Classification: Multimodal Phenotyping in a Clinical Psychiatric Cohort},
  author    = {Li, Siyu and Perkoff, E. Margaret and Ungar, Lyle},
  booktitle = {[BOOKTITLE]},
  year      = {2026},
  url       = {[PAPER_URL]}
}

@inproceedings{Hicks2021,
  title     = {{HYPERAKTIV}: An Activity Dataset from Patients with Attention-Deficit/Hyperactivity Disorder ({ADHD})},
  author    = {Hicks, Steven A. and Stautland, Andrea and Fasmer, Ole Bernt and F{\o}rland, Wenche and
               Hammer, Hugo Lewi and Halvorsen, P{\aa}l and Mjeldheim, Kristin and Oedegaard, Ketil Joachim and
               Osnes, Berge and Syrstad, Vigdis Elin Gi{\ae}ver and Riegler, Michael Alexander and Jakobsen, Petter},
  booktitle = {Proceedings of the 12th ACM Multimedia Systems Conference (MMSys '21)},
  pages     = {314--319},
  year      = {2021},
  doi       = {10.1145/3458305.3478454}
}
```

## License

The code in this repository is released under the MIT License (see [LICENSE](LICENSE)). The HYPERAKTIV dataset is not part of this repository and is subject to its own terms (see [Data](#data)).
