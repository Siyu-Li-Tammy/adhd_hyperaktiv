# HYPERAKTIV-ML: Revised Paper Sections
## Rethinking ADHD Classification: A Multimodal Analysis of Cognitive, Behavioral, and Physiological Signals

---

## RESPONSE TO YOUR 7 POINTS

---

## 1. PERMUTATION TEST p-VALUES — ARE THEY ALL p=0.000 VALID?

**Short answer: Mostly yes, but two need nuance in the paper.**

From your notebook outputs, the binary classification permutation p-values (500 iterations, balanced accuracy) are:

| Modality Block | n | AUC | perm_p | Valid? |
|---|---|---|---|---|
| CPT + Self-report | 97 | 0.895 | 0.002 | ✅ Strong |
| Self-report only | 94 | 0.851 | 0.002 | ✅ Strong |
| CPT only | 97 | 0.800 | 0.002 | ✅ Strong |
| CPT + HRV | 75 | 0.785 | 0.002 | ✅ Strong |
| All modalities | 64 | 0.750 | 0.006 | ✅ Strong |
| CPT + Activity | 79 | 0.706 | 0.002 | ✅ Strong |
| CPT + Act + HRV | 64 | 0.671 | 0.044 | ✅ Marginal |
| Activity only | 85 | 0.528 | 0.611 | ❌ Not significant |
| HRV only | 80 | 0.378 | 0.798 | ❌ Not significant |

**Why p=0.002 (not 0.000)?** With 500 permutation iterations, p=0.002 = 1/500, meaning the observed stat exceeded ALL permuted stats. This is the minimum achievable p-value with 500 iterations — it is NOT p=0.000. The paper should report these as p=0.002 (not p<0.001 or 0.000) unless you ran more iterations.

**Key issue to address in paper:** CPT + Act + HRV has p=0.044, which is marginal at α=0.05. With multiple testing across 9 blocks, this may not survive FDR correction. Consider noting this or applying Bonferroni (threshold = 0.05/9 = 0.0056), which would make this block non-significant.

**For clustering permutation tests:** All 8 modality blocks show p=0.0000 (with 500 iterations, meaning p ≤ 0.002). These are the silhouette score permutations, not classification accuracy. They test whether geometric cluster quality exceeds chance geometry — valid, but note this tests cluster *structure*, not clinical relevance (the low ARI values confirm clusters don't map to clinical labels even when geometrically sound).

---

## 2. REVISED INTRODUCTION (Condensed Related Work, Pointed at Your Contributions)

### 1 Introduction

Attention-Deficit/Hyperactivity Disorder (ADHD) affects an estimated 5–10% of children and 2–5% of adults worldwide (Prasad and Kumminimana, 2025), yet clinical diagnosis remains heavily dependent on subjective symptom reports and clinician judgment — methods that are context-sensitive and vulnerable to reporting bias. Objective neuropsychological assessment, in particular the Conners Continuous Performance Test II (CPT-II), offers a task-based measure of sustained attention and inhibitory control, and prior work using gradient boosting on CPT-derived features has achieved AUC > 0.90 for ADHD classification (Casals et al., 2025), motivating its central role in our pipeline.

Despite strong CPT-II performance, the field has largely defaulted to binary ADHD/non-ADHD classification, obscuring clinically meaningful heterogeneity. Neuropsychological subtyping studies identify multiple distinct cognitive profiles in adult ADHD — including impaired attention/inhibition, elevated delay discounting, and working memory deficits (Mostert et al., 2018) — suggesting that clustering may better capture clinical reality than a binary label. Yet no established method exists for recovering ADHD subtypes (inattentive/ADD, hyperactive-impulsive, combined) from ML outputs.

Beyond cognitive testing, wearable-derived actigraphy and heart rate variability (HRV) have been explored as complementary modalities for ADHD sensing. Temporal segmentation of actigraphy by time-of-day improves classification, suggesting behavioral rhythmicity — not raw activity — as the discriminative signal (V and S. D., 2024). Self-report questionnaires, particularly the ASRS and Conners' CAARS, carry strong diagnostic signal in isolation; a LightGBM model trained on CAARS items achieved competitive accuracy across 1,629 adults, though performance degraded when distinguishing ADHD from other psychiatric conditions (Christiansen et al., 2020) — a finding directly relevant to our setting.

A critical and underappreciated challenge is control group composition. The HYPERAKTIV dataset (Hicks et al., 2021) uses psychiatric controls — adults with comorbid mood, anxiety, and substance-use disorders — rather than healthy volunteers. This design is more clinically realistic but substantially harder for physiological classifiers, since autonomic dysregulation and motor irregularities are shared across these conditions. Prior work on the same dataset typically does not address this confound explicitly.

We present HYPERAKTIV-ML, a unified multimodal pipeline that integrates four biological modalities — CPT-II, HRV, wrist actigraphy, and self-report questionnaires — across nine modality combinations. Our contributions are: (1) a systematic incremental analysis establishing which modalities add diagnostic value beyond CPT-II alone; (2) a 3-way classification framework separating ADD, combined ADHD, and non-ADHD that exposes ADD as the bottleneck for all classifiers; and (3) per-modality unsupervised clustering demonstrating that different modalities capture independent biological dimensions, and that cognitive subtype structure is better recovered than binary diagnostic labels across all modalities.

---

## 2. REVISED RELATED WORK (Combined 2.2+2.3, Tightened 2.3)

### 2 Related Work

#### 2.1 CPT-Based Classification

Machine learning applied to CPT-II has consistently achieved strong ADHD classification performance, with gradient boosting and LightGBM reaching AUC > 0.90 on pediatric and adult samples (Casals et al., 2025). Key predictive features — omission errors, commission errors, reaction time variability, and the d′ signal detection index — closely parallel the manually defined ADHD clinical fingerprint. A prior study on the HYPERAKTIV dataset demonstrated that LightGBM can forecast standardized CPT-II T-scores and the ADHD Confidence Index from CPT raw scores and basic demographics (Satvik et al., 2025), confirming the feasibility of supervised learning on this data format and informing our feature engineering choices.

#### 2.2 Physiological, Wearable, and Self-Report Modalities

Actigraphy-based ADHD classification on HYPERAKTIV has yielded moderate performance across LR, RF, XGBoost, and LightGBM classifiers under 10-fold cross-validation; temporal segmentation by time-of-day (morning/noon/evening/night) improves accuracy, suggesting behavioral rhythmicity as the more discriminative signal over raw activity level (V and S. D., 2024). A situation-aware approach integrating HRV and activity features retained 60 of 1,576 engineered features and found Random Forest as best classifier, underscoring that time-sensitive, context-aware feature design outperforms naive aggregation (Hicks et al., 2021). Network analysis on HYPERAKTIV's wearable data identified five community structures aligning with distinct psychiatric disorder profiles, providing motivation for unsupervised clustering alongside supervised classification (Mostert et al., 2018).

Self-report questionnaires, particularly ASRS and Conners' CAARS, carry strong classification signal: a LightGBM model trained on 26 CAARS items achieved competitive diagnostic accuracy across a 1,629-subject cohort including ADHD, obesity, gambling disorder, and healthy controls, though performance degraded markedly when distinguishing ADHD from other psychiatric conditions (Christiansen et al., 2020). This degradation foreshadows our multimodal fusion results — when controls share symptom profiles with ADHD patients, self-report gains discriminative edge over physiological modalities, which cannot separate overlapping autonomic profiles. We extend these single-modality findings by systematically quantifying the marginal diagnostic contribution of each modality beyond CPT-II, a gap not previously addressed in the HYPERAKTIV literature.

#### 2.3 ADHD Subtyping and Computational Heterogeneity

Neuropsychological subtyping studies identify multiple cognitive profiles in adults with ADHD — including combined attention/inhibition impairment, elevated delay discounting, and working memory deficits — and demonstrate that similar subgroups emerge in both ADHD and healthy control populations, challenging purely categorical diagnosis (Mostert et al., 2018). Recent automated ADHD detection reviews confirm that the field has largely defaulted to binary classification with no established computational method for recovering inattentive, hyperactive, or combined subtypes from ML outputs (Olinic et al., 2025). We address this gap directly through unsupervised K-Means and agglomerative clustering on CPT-II T-score profiles, evaluating alignment with the three-class ADHD/ADD/non-ADHD diagnostic structure.

---

## 3. SECTION 3 — EDA / CPT-II ANALYSIS ADDITION

### 3.3 CPT-II Exploratory Analysis

**Replicating the Classic ADHD CPT Profile.** Raw CPT-II metrics successfully replicate the established ADHD neuropsychological fingerprint. Five features survived Benjamini-Hochberg FDR correction (α = 0.05) across the 12-feature multiple-testing scenario, with medium-to-large rank-biserial effect sizes (Mann-Whitney U): omission errors (r = 0.42, p_fdr = 0.003), commission errors (r = 0.51, p_fdr < 0.001), hit RT standard error (r = 0.45, p_fdr = 0.002), variability of SE (r = 0.51, p_fdr < 0.001), d′ (r = −0.43, p_fdr = 0.002), and perseverations (r = 0.41, p_fdr = 0.005). The expected directional pattern — higher omissions (inattention) and commissions (impulsivity) in ADHD — is confirmed. Reaction time itself did not survive correction, consistent with an adult sample skewed toward the inattentive rather than hyperactive presentation.

**Distribution Overlap.** Despite statistical separation, CPT-II distributions overlap substantially between ADHD and non-ADHD groups across all features. This overlap motivates fusion with complementary modalities and explains why a modest number of features (omissions, commissions, d′) drive most classification signal — they maximize separation within an otherwise overlapping space.

**T-Score Standardization Choice.** For classification, raw scores are used: T-scores normalize toward the ADHD population, which inflates apparent diagnostic separation. For clustering, T-scores are used: placing all CPT-II dimensions on a common scale (mean = 50, SD = 10) is necessary for Euclidean-distance-based K-Means to treat features equitably.

**Dataset Composition.** Table 1 summarizes participant coverage per modality. Note that the all-modality fusion subset (n = 64) is substantially reduced from the full cohort (n = 103), reflecting the real-world constraint that not all patients complete all assessments — a limitation we return to in Section 7.

---

## 4. REVISED SECTION 4.1 METRICS (Prose, with Citations)

### 4.1 Binary Classification

#### Classifiers and Validation

All classifiers from the original HYPERAKTIV benchmark — Logistic Regression (LR), Random Forest (RF), XGBoost, and LightGBM — were evaluated under 10-fold stratified cross-validation to preserve class balance across splits. Dummy classifiers (random, minority-class, majority-class) are included as baselines.

Statistical significance was confirmed per modality block via a 500-iteration permutation test on observed balanced accuracy, guarding against inflated performance from random label–feature correlations.

#### Metrics

We report five complementary metrics following established practice in clinical ML and ADHD classification literature. ROC-AUC serves as the primary threshold-independent ranking metric, widely adopted in ADHD classification benchmarks (Casals et al., 2025; Christiansen et al., 2020) because it evaluates classifier discrimination independently of decision threshold. Balanced accuracy (arithmetic mean of sensitivity and specificity) is reported as the primary performance scalar under class imbalance, preferred over raw accuracy when class distributions are unequal (Brodersen et al., 2010). The Matthews Correlation Coefficient (MCC) is included as a single balanced metric robust to class imbalance that captures all four confusion matrix cells; it is recommended over F1 for binary classification when both classes matter clinically (Chicco and Jurman, 2020). Precision, recall, and F1 round out the reported set. For individual CPT-II features, rank-biserial r (the nonparametric effect size for Mann-Whitney U) is computed, with p-values adjusted using Benjamini-Hochberg FDR correction to control the false discovery rate across the 12-feature multiple-testing scenario.

---

## 5. REVISED RESULTS STRUCTURE

**[See note: each experiment should be: 1 table + 1–2 paragraph summary. Move extended interpretation to Section 6 Discussion.]**

### 5.1 Binary Classification Results

[TABLE: Binary classification — best model per modality block, ranked by AUC. See Table 2 in paper.]

CPT combined with self-report achieves the best binary classification performance (AUC = 0.895, balanced accuracy = 81.6%, F1 = 0.813, MCC = 0.632, permutation p = 0.002, n = 97), representing a +9.4 AUC-point gain over CPT alone (AUC = 0.800). All CPT-containing modality blocks reach statistical significance (p ≤ 0.044). In contrast, HRV alone (AUC = 0.378, p = 0.798) and actigraphy alone (AUC = 0.528, p = 0.611) both fail to exceed chance, consistent with the hypothesis that the psychiatric control group's comorbid mood and anxiety disorders produce overlapping autonomic and motor signatures indistinguishable from ADHD.

### 5.2 3-Way Classification Results

[TABLE: 3-way classification — macro-F1, macro-AUC, balanced accuracy, per-class recall. See Table 3.]

Self-report alone achieves the best 3-way performance (macro-F1 = 0.632, macro-AUC = 0.804, balanced accuracy = 64.4%, permutation p = 0.003). ADD recall is the universal bottleneck: the best model achieves ADHD recall = 0.800 and non-ADHD recall = 0.739 but ADD recall = only 0.391. The gap between macro-AUC and macro-F1 across all modalities reveals that models can correctly rank patients along a severity axis (AUC captures ordering) but struggle to place clean categorical boundaries (F1 penalizes discrete misclassification).

### 5.3 Clustering Results

[TABLE: Per-modality clustering — k, silhouette, DB, ARI vs ADHD, ARI vs CPT subtypes, χ² p. See Table 4.]

CPT-II K-Means (k = 4, silhouette = 0.236, DB = 1.367) identifies a cognitive severity gradient: a low-impairment cluster (40% ADHD, moderate errors), an intermediate combined profile (71% ADHD, elevated commission and omission errors), and two ADD-like clusters (high omissions, low commissions). The moderate silhouette score indicates continuous rather than categorically distinct impairment dimensions. HRV achieves the highest silhouette score (0.309, k = 3) but near-zero ARI with clinical labels (ARI = −0.008), indicating geometrically clean but clinically irrelevant clusters. CPT + self-report fusion achieves the best cross-modal subtype recovery (ARI vs CPT subtypes = 0.198, k = 4). All modalities show higher ARI against CPT-defined subtypes than against binary ADHD diagnosis labels, supporting unsupervised clustering as a cognitive phenotyping tool rather than a diagnostic proxy.

---

## 6. NEW DISCUSSION SECTION

### 6 Discussion

#### 6.1 Self-Report as the Key Complementary Modality

The +9.4 AUC-point gain from adding self-report questionnaires to CPT-II reflects non-redundant signal: CPT-II captures objective cognitive performance deficits (omissions, commissions, d′), while ASRS, WURS, MADRS, and HADS capture subjective symptom burden, childhood history, and mood-anxiety comorbidity. The psychiatric control group — patients with mood, anxiety, and substance-use disorders — is behaviorally distinct from ADHD on self-report but not on physiological measures, explaining why self-report provides discriminative lift while HRV and actigraphy do not.

#### 6.2 Physiological Modalities and the Psychiatric Control Problem

The failure of HRV (AUC = 0.378) and actigraphy (AUC = 0.528) as standalone classifiers is not simply a data quality issue — both show well-structured clusters geometrically (silhouette > 0.13) — but rather a signal contamination issue. Autonomic dysregulation is a shared feature of ADHD, anxiety, and mood disorders; motor irregularity is common across psychiatric conditions. This finding directly argues that physiological classifiers benchmarked against healthy controls will systematically overstate real-world clinical utility. Future ADHD assessment systems should treat control group composition as an experimental design variable.

#### 6.3 ADD as the Classification Bottleneck

ADD recall consistently lags 30–40 percentage points behind ADHD and non-ADHD recall across all modalities. This is not a model failure but a signal problem: ADD presents on a continuum with both ADHD (shared inattention) and non-ADHD (absent impulsivity overlap), without a boundary that is clean in any single modality space. The high macro-AUC relative to macro-F1 confirms that models can rank ADD patients correctly — they receive intermediate scores — but cannot place a discrete decision boundary around them.

#### 6.4 Clustering Captures Subtype Structure, Not Diagnostic Labels

The consistent pattern that all modalities show higher ARI against CPT-defined subtypes (0.005–0.198) than against binary ADHD diagnosis labels (−0.016–0.179) suggests that unsupervised clustering recovers finer cognitive structure that the binary label masks. The ADHD-only clustering (silhouette = 0.353) demonstrates that the diagnosed group is internally heterogeneous in a geometrically meaningful way, even though this structure does not map cleanly onto ADD vs. combined ADHD subtype labels. This supports treating cluster membership as a research endpoint in itself — a cognitive phenotype, not a diagnostic category. HRV's anomaly (high silhouette = 0.309 but ARI ≈ 0) is particularly instructive: it captures physiological groupings real enough to produce geometric separation, but those groupings track something orthogonal to ADHD diagnosis — possibly circadian rhythm disruption shared across all psychiatric groups.

---

## 7. REVISED CONCLUSION

### 7 Conclusion

We presented HYPERAKTIV-ML, a unified multimodal pipeline applied to the HYPERAKTIV clinical dataset integrating four biological modalities — CPT-II, HRV, wrist actigraphy, and five validated self-report questionnaires — across nine modality combinations evaluated through binary classification, 3-way multiclass classification, and per-modality unsupervised clustering.

Our primary finding is that CPT-II combined with self-report questionnaires (ASRS, WURS, MADRS, HADS-A/D) achieves the strongest ADHD binary classification performance (AUC = 0.895, balanced accuracy = 81.6%), with objective cognitive performance and subjective symptom burden providing complementary, non-redundant diagnostic signal. Physiological modalities — HRV and actigraphy — fail to reach statistical significance as standalone classifiers against a psychiatric control group, a finding we attribute to shared autonomic and motor dysregulation across the diagnostic conditions present in the control cohort.

In 3-way classification, ADD subtype recall (≈39%) consistently lags non-ADHD and combined ADHD recall across all modalities, reflecting the absence of a clean signal boundary for inattentive-only presentations rather than model inadequacy. The gap between high macro-AUC and lower macro-F1 confirms that models can correctly rank ADD patients on a severity axis without being able to place discrete category boundaries around them.

Unsupervised CPT-II clustering reveals a 4-cluster cognitive severity gradient rather than a binary ADHD/non-ADHD split, supporting a dimensional rather than categorical view of adult ADHD. Crucially, all modalities achieve higher adjusted rand index against CPT-defined cognitive subtypes than against binary ADHD diagnosis labels, suggesting that clustering is better suited as a cognitive phenotyping tool than as a diagnostic proxy.

Two implications are significant for the field. First, future ADHD assessment systems should treat the control group composition as an experimental design variable: benchmarking against healthy volunteers systematically overstates the real-world utility of physiological classifiers, and clinical-control comparisons like HYPERAKTIV provide a more honest evaluation. Second, the sample-size versus modality-richness tradeoff — complete four-modality overlap reducing n from 97 to 64 — limits conclusions about all-modality fusion and points to the need for larger, prospectively designed multimodal ADHD cohorts.

Future work should prioritize two directions: larger multimodal ADHD datasets with psychiatric controls to validate the complementarity of CPT-II and self-report under realistic clinical conditions, and longitudinal actigraphy and HRV collection that allows rhythmicity features (circadian amplitude, intra-daily variability) to be estimated with greater precision — the modalities that appear weakest here may yet carry signal recoverable only with richer temporal data.

---

## APPENDIX NOTE: Self-assessment features used

Five validated instruments were included: ASRS (Adult ADHD Self-Report Scale, 6 items), WURS (Wender Utah Rating Scale, 25 items, childhood ADHD retrospective), MADRS (Montgomery-Åsberg Depression Rating Scale, 10 items), HADS-A (Hospital Anxiety and Depression Scale — Anxiety, 7 items), and HADS-D (Hospital Anxiety and Depression Scale — Depression, 7 items). Composite scores per instrument were used rather than item-level features to prevent overfitting given n = 103.
