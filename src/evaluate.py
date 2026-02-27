from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix, roc_curve, precision_recall_curve,
    brier_score_loss
)
from sklearn.calibration import calibration_curve


def _get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    raise ValueError("Model has neither predict_proba nor decision_function")


def compute_metrics(model, X, y, threshold: float = 0.5) -> dict:
    scores = _get_scores(model, X)
    y_pred = (scores >= threshold).astype(int)

    roc = roc_auc_score(y, scores)
    pr = average_precision_score(y, scores)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y, y_pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    # calibration proxy
    try:
        brier = brier_score_loss(y, np.clip(scores, 0, 1))
    except Exception:
        brier = None

    return {
        "threshold": float(threshold),
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "brier": None if brier is None else float(brier),
    }


def save_metrics(metrics: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def plot_confusion(y_true, y_scores, threshold: float, title: str, out_path: Path):
    y_pred = (y_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    fig = plt.figure(figsize=(5, 4))
    plt.imshow([[tn, fp], [fn, tp]])
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    plt.title(title)
    for (i, j), v in np.ndenumerate([[tn, fp], [fn, tp]]):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_roc(y_true, y_scores, title: str, out_path: Path):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fig = plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_pr(y_true, y_scores, title: str, out_path: Path):
    p, r, _ = precision_recall_curve(y_true, y_scores)
    fig = plt.figure(figsize=(5, 4))
    plt.plot(r, p)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_calibration(y_true, y_scores, title: str, out_path: Path, n_bins: int = 10):
    # only meaningful if scores are probabilities; if decision_function, still ok but interpret with caution
    y_scores = np.clip(y_scores, 0, 1)
    frac_pos, mean_pred = calibration_curve(y_true, y_scores, n_bins=n_bins, strategy="uniform")

    fig = plt.figure(figsize=(5, 4))
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0, 1], [0, 1])
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(title)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def evaluate_and_save(model, X, y, model_name: str, fig_dir: Path, met_dir: Path, threshold: float = 0.5):
    scores = _get_scores(model, X)
    metrics = compute_metrics(model, X, y, threshold=threshold)
    metrics["model"] = model_name
    save_metrics(metrics, met_dir / f"{model_name}_metrics.json")

    plot_roc(y, scores, f"ROC: {model_name}", fig_dir / f"{model_name}_roc.png")
    plot_pr(y, scores, f"PR: {model_name}", fig_dir / f"{model_name}_pr.png")
    plot_confusion(y, scores, threshold, f"Confusion @ {threshold:.2f}: {model_name}",
                   fig_dir / f"{model_name}_confusion.png")
    plot_calibration(y, scores, f"Calibration: {model_name}", fig_dir / f"{model_name}_calibration.png")

    return metrics