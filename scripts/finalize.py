from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    brier_score_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.inspection import permutation_importance

# EBM
from interpret.glassbox import ExplainableBoostingClassifier


RANDOM_STATE = 42

# processed data created in Step 3
X_TRAIN_PATH = Path("data/processed/X_train.csv")
Y_TRAIN_PATH = Path("data/processed/y_train.csv")
X_VAL_PATH   = Path("data/processed/X_val.csv")
Y_VAL_PATH   = Path("data/processed/y_val.csv")
X_TEST_PATH  = Path("data/processed/X_test.csv")
Y_TEST_PATH  = Path("data/processed/y_test.csv")

FINAL_MODEL_NAME = "EBM_Final"
# Use the validated operating threshold you already selected/tuned on validation:
FINAL_THRESHOLD = 0.11


def project_root_from_file() -> Path:
    return Path(__file__).resolve().parents[1]

def make_run_paths(root: Path, tag: str) -> Dict[str, Path]:
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = root / "outputs" / "runs" / f"{tag}_{ts}"
    met_dir = run_dir / "metrics"
    fig_dir = run_dir / "figures"
    mod_dir = run_dir / "models"
    run_dir.mkdir(parents=True, exist_ok=True)
    met_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    mod_dir.mkdir(parents=True, exist_ok=True)
    return {"run_dir": run_dir, "met_dir": met_dir, "fig_dir": fig_dir, "mod_dir": mod_dir}

def load_processed_xy(root: Path) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X_train = pd.read_csv(root / X_TRAIN_PATH)
    y_train = pd.read_csv(root / Y_TRAIN_PATH).iloc[:, 0]
    X_val = pd.read_csv(root / X_VAL_PATH)
    y_val = pd.read_csv(root / Y_VAL_PATH).iloc[:, 0]
    X_test = pd.read_csv(root / X_TEST_PATH)
    y_test = pd.read_csv(root / Y_TEST_PATH).iloc[:, 0]
    return X_train, y_train, X_val, y_val, X_test, y_test

def infer_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "string", "bool"]).columns.tolist()

    # drop ID-like
    for id_col in ["customer_id", "id", "customerID"]:
        if id_col in num_cols: num_cols.remove(id_col)
        if id_col in cat_cols: cat_cols.remove(id_col)

    return num_cols, cat_cols

def build_preprocess(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(
        [("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
        remainder="drop",
    )

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Dict[str, Any]:
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "threshold": float(thr),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }

def save_eval_plots(fig_dir: Path, name: str, y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> None:
    # ROC
    plt.figure(figsize=(6,5))
    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.title(f"{name} — ROC (test)")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{name}_roc_test.png")
    plt.close()

    # PR
    plt.figure(figsize=(6,5))
    PrecisionRecallDisplay.from_predictions(y_true, y_prob)
    plt.title(f"{name} — PR (test)")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{name}_pr_test.png")
    plt.close()

    # Confusion
    y_pred = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    plt.imshow(cm)
    plt.title(f"{name} — Confusion (thr={thr:.2f})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{name}_confusion_test.png")
    plt.close()

def latest_step5_params(root: Path) -> Dict[str, Any]:
    runs_dir = root / "outputs" / "runs"
    step5 = sorted(runs_dir.glob("step5_*"), key=lambda p: p.stat().st_mtime)
    if not step5:
        return {}
    latest = step5[-1]
    csv_path = latest / "metrics" / "tuned_comparison_val.csv"
    if not csv_path.exists():
        return {}

    df = pd.read_csv(csv_path)
    row = df[df["model"] == "EBM_Tuned"].iloc[0]
    bp = row.get("best_params", None)
    if isinstance(bp, str) and len(bp) > 0:
        try:
            return json.loads(bp)
        except Exception:
            return {}
    return {}

def permutation_driver_importance(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    out_csv: Path,
    out_png: Path,
    n_repeats: int = 5,
) -> pd.DataFrame:
    # permute original columns; scoring aligned with imbalance
    r = permutation_importance(
        model, X_test, y_test,
        scoring="average_precision",
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    imp = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": r.importances_mean,
        "importance_std": r.importances_std,
    }).sort_values("importance_mean", ascending=False)

    imp.to_csv(out_csv, index=False)

    top = imp.head(20).iloc[::-1]
    plt.figure(figsize=(8,6))
    plt.barh(top["feature"], top["importance_mean"])
    plt.title("Top drivers (Permutation Importance on test; PR-AUC drop)")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    return imp


def main():
    root = project_root_from_file()
    sys.path.insert(0, str(root))

    run = make_run_paths(root, "step6")
    met_dir, fig_dir, mod_dir = run["met_dir"], run["fig_dir"], run["mod_dir"]

    X_train, y_train, X_val, y_val, X_test, y_test = load_processed_xy(root)

    # retrain on train+val
    X_tr = pd.concat([X_train, X_val], ignore_index=True)
    y_tr = pd.concat([y_train, y_val], ignore_index=True)

    num_cols, cat_cols = infer_feature_types(X_tr)
    preprocess = build_preprocess(num_cols, cat_cols)

    # Build final EBM; load tuned params if available
    ebm = ExplainableBoostingClassifier(random_state=RANDOM_STATE)
    pipe = Pipeline([("preprocess", preprocess), ("clf", ebm)])

    params = latest_step5_params(root)
    # best_params keys are like "clf__max_bins" etc.
    if params:
        pipe.set_params(**params)

    pipe.fit(X_tr, y_tr)

    # test evaluation
    y_prob = pipe.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test.values, y_prob, FINAL_THRESHOLD)
    metrics["model"] = FINAL_MODEL_NAME
    metrics["train_rows"] = int(len(X_tr))
    metrics["test_rows"] = int(len(X_test))

    (met_dir / "final_test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # save test predictions
    pred_df = X_test.copy()
    pred_df["y_true"] = y_test.values
    pred_df["y_prob"] = y_prob
    pred_df["y_pred"] = (y_prob >= FINAL_THRESHOLD).astype(int)
    pred_df.to_csv(met_dir / "final_test_predictions.csv", index=False)

    # plots
    save_eval_plots(fig_dir, FINAL_MODEL_NAME, y_test.values, y_prob, FINAL_THRESHOLD)

    # save model artifact
    joblib.dump(pipe, mod_dir / f"{FINAL_MODEL_NAME}.joblib")

    # driver analysis
    permutation_driver_importance(
        model=pipe,
        X_test=X_test,
        y_test=y_test,
        out_csv=met_dir / "driver_importance_permutation.csv",
        out_png=fig_dir / "driver_importance_top20.png",
        n_repeats=5,
    )

    # minimal run summary
    pd.DataFrame([metrics]).to_csv(met_dir / "final_test_metrics.csv", index=False)
    print("Saved Step6 run to:", run["run_dir"])
    print("Test metrics:", metrics)


if __name__ == "__main__":
    main()