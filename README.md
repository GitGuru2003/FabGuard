# FabGuard

**Semiconductor manufacturing failure detection from sensor data**
Munib Ahmed & Sindi Banda · CS 533 Introduction to Data Science · Fall 2026

FabGuard studies whether high-dimensional process sensor data can separate passing from failing
semiconductor production runs. It uses the [UCI SECOM dataset](https://archive.ics.uci.edu/dataset/179/secom):
1,567 samples, 590 sensors, and only 104 failures (~6.6%).

We compare supervised classifiers (Logistic Regression, XGBoost) against unsupervised anomaly
detection (Isolation Forest) with a Dummy baseline. We also look at how class-imbalance handling
(class weights, SMOTE) and decision thresholds trade off recall against false alarms. PR-AUC and
recall are the main metrics.

## Project structure

```
FabGuard/
├── Data/
│   ├── Raw/              # Original SECOM files (secom.data, secom_labels.data, secom.names)
│   └── Processed/        # Generated intermediate data (git-ignored)
├── Notebooks/
│   └── 01_eda.ipynb      # Exploratory analysis; writes Reports/Figures/eda_*.png
├── src/FabGuard/         # Reusable project code
│   ├── config.py         # Paths, random seed, split/CV settings
│   ├── data.py           # Load SECOM + labels + timestamps, stratified split
│   ├── preprocessing.py  # Leakage-safe pipeline: missingness filter, imputation, scaling, selection
│   ├── models.py         # Dummy, Logistic Regression, Isolation Forest, XGBoost builders
│   ├── evaluation.py     # Metrics, threshold selection, confusion matrix / PR curves
│   └── plotting.py       # Shared figure style and colors
├── Scripts/
│   ├── train.py          # Train + tune models with stratified CV
│   └── evaluate.py       # Final test-set evaluation, writes to Reports/
├── Models/               # Saved models (git-ignored)
├── Reports/
│   ├── Figures/          # Generated plots (git-ignored)
│   └── Tables/           # Generated metric tables (git-ignored)
├── Tests/                # pytest tests
├── pyproject.toml
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                   # makes `import FabGuard` work from notebooks and scripts
```

Run the tests with `pytest` and the training script with `python Scripts/train.py`
(or `--models dummy logreg` for a subset; all four models take ~35 s). Training uses nested
stratified CV on the training split only and writes:

- `Reports/Tables/cv_results.csv`: precision, recall, specificity, F1, GM, MCC, ROC-AUC, PR-AUC per model, at the
  default threshold and at the F1-maximizing threshold
- `Reports/Tables/cv_oof_scores.csv`: out-of-fold scores, for further plots
- `Reports/Figures/cv_pr_curves.png`, `cv_confusion_matrices.png`
- `Models/<model>.joblib`: refit model, chosen threshold, and best hyperparameters

To regenerate the EDA figures, run the notebook top to bottom, or:

```bash
jupyter nbconvert --to notebook --execute --inplace Notebooks/01_eda.ipynb
```

## Key EDA findings

- 6.6% failures; always predicting "pass" is 93.4% accurate.
- The failure rate drifts: 10.5% before Sep 1, 2008 vs 4.1% after.
- 24 sensors are over 50% missing and 116 are constant (training split). Gaps occur in time blocks
  and barely predict failure.
- The signal is weak: 15 sensors survive FDR correction, and the strongest |r| with failure is 0.17.
- About half the sensors are near-duplicates (|r| > 0.8). PCA shows no class separation, and PC2
  tracks time.

## Methodology rules

- **Split first.** Call `split_data` before fitting any imputer, scaler, selector, or resampler.
- **Fit on training data only.** Preprocessing and SMOTE go inside the model pipeline so they run
  within each CV fold.
- **Keep the test set out of model selection.** Hyperparameters and thresholds are chosen with
  stratified CV on the training split.
- Use `config.RANDOM_STATE` everywhere for reproducibility.

## Timeline

| Dates | Phase |
|---|---|
| Sep 14 – Sep 27 | Data setup and exploration |
| Sep 28 – Oct 11 | Preprocessing and feature reduction |
| Oct 12 – Oct 22 | Dummy and Logistic Regression baselines |
| Oct 23 | **Midterm presentation**; train Isolation Forest and XGBoost |
| Oct 24 – Nov 13 | Compare imbalance handling and thresholds |
| Nov 14 – Nov 27 | Model evaluation |
| Nov 28 – Dec 8 | Results, visualizations, and tables |
| Dec 11 | **Final presentation** |
