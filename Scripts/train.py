"""Train and cross-validate models on the training split; save models, CV metrics and figures.

Uses nested stratified CV: each outer fold tunes hyperparameters with an inner
GridSearchCV (scored by PR-AUC), then scores the held-out outer fold. The pooled
out-of-fold scores give the CV metrics and are used to pick each model's decision
threshold. The test split is never touched here; see Scripts/evaluate.py.

Usage:
    python Scripts/train.py                          # all models
    python Scripts/train.py --models dummy logreg    # a subset
"""

import argparse
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from FabGuard import config
from FabGuard.data import load_secom, split_data
from FabGuard.evaluation import (
    compute_metrics,
    get_scores,
    plot_confusion_matrix,
    plot_precision_recall_curves,
    select_threshold,
)
from FabGuard.models import model_specs
from FabGuard.plotting import MODEL_NAMES, set_style

DEFAULT_THRESHOLD = 0.5


def make_search(pipeline, grid, cv):
    if not grid:
        return pipeline
    return GridSearchCV(pipeline, grid, scoring="average_precision", cv=cv, n_jobs=-1)


def out_of_fold_scores(pipeline, grid, X, y) -> np.ndarray:
    """Nested CV: out-of-fold failure scores for every training sample."""
    outer = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    inner = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE + 1)
    scores = np.empty(len(y))
    for train_idx, val_idx in outer.split(X, y):
        search = make_search(pipeline, grid, inner)
        search.fit(X.iloc[train_idx], y.iloc[train_idx])
        scores[val_idx] = get_scores(search, X.iloc[val_idx])
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", help="Subset of models to train (default: all)")
    args = parser.parse_args()

    set_style()
    X, y, _ = load_secom()
    X_train, _, y_train, _ = split_data(X, y)
    print(f"Train: {len(y_train)} samples ({y_train.sum()} failures)\n")

    specs = model_specs(y_train)
    names = args.models or list(specs)
    unknown = set(names) - set(specs)
    if unknown:
        parser.error(f"unknown models {sorted(unknown)}; choose from {list(specs)}")

    for d in (config.MODELS_DIR, config.TABLES_DIR, config.FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    rows, oof = [], {}
    for name in names:
        pipeline, grid = specs[name]
        start = time.time()
        scores = out_of_fold_scores(pipeline, grid, X_train, y_train)
        oof[name] = scores

        # The dummy's constant score has no meaningful threshold to tune
        threshold = DEFAULT_THRESHOLD if name == "dummy" else select_threshold(y_train, scores)
        rows.append({"model": name, "threshold_rule": "default", **compute_metrics(y_train, scores, DEFAULT_THRESHOLD)})
        rows.append({"model": name, "threshold_rule": "max_f1", **compute_metrics(y_train, scores, threshold)})

        # Refit on the full training split for later test-set evaluation
        final = make_search(pipeline, grid, StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE))
        final.fit(X_train, y_train)
        best_params = getattr(final, "best_params_", {})
        joblib.dump(
            {"model": final, "threshold": threshold, "best_params": best_params},
            config.MODELS_DIR / f"{name}.joblib",
        )
        print(f"{name:<17} {time.time() - start:5.1f}s  best params: {best_params}")

    results = pd.DataFrame(rows)
    results.to_csv(config.TABLES_DIR / "cv_results.csv", index=False)
    pd.DataFrame(oof).assign(fail=y_train.to_numpy()).to_csv(config.TABLES_DIR / "cv_oof_scores.csv", index=False)

    ax = plot_precision_recall_curves({n: (y_train, s) for n, s in oof.items() if n != "dummy"})
    ax.set_title("Out-of-fold precision-recall (training CV)")
    ax.figure.savefig(config.FIGURES_DIR / "cv_pr_curves.png", dpi=150, bbox_inches="tight")

    fig, axes = plt.subplots(1, len(oof), figsize=(3.2 * len(oof), 3.2))
    tuned = results[results.threshold_rule == "max_f1"].set_index("model")
    for ax, (n, s) in zip(np.atleast_1d(axes), oof.items()):
        plot_confusion_matrix(y_train, (s >= tuned.loc[n, "threshold"]).astype(int), ax=ax, title=MODEL_NAMES[n])
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "cv_confusion_matrices.png", dpi=150, bbox_inches="tight")

    cols = ["model", "threshold_rule", "accuracy", "precision", "recall", "specificity", "f1", "gm", "mcc", "roc_auc", "pr_auc"]
    print("\nCross-validated metrics on the training split:")
    print(results[cols].round(3).to_string(index=False))
    print(f"\nSaved models to {config.MODELS_DIR}, tables to {config.TABLES_DIR}, figures to {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
