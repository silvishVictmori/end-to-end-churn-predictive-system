from __future__ import annotations

import json
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
import joblib

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from interpret.glassbox import ExplainableBoostingClassifier
    HAS_EBM = True
except Exception:
    HAS_EBM = False



RANDOM_STATE = 42
N_ITER = 25          
CV = 3               # CV folds
OBJECTIVE = "f1"     
MIN_PRECISION = 0.25 

# processed CSVs created in Step 3
X_TRAIN_PATH = Path("data/processed/X_train.csv")
Y_TRAIN_PATH = Path("data/processed/y_train.csv")
X_VAL_PATH   = Path("data/processed/X_val.csv")
Y_VAL_PATH   = Path("data/processed/y_val.csv")


# Utilities
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

def load_processed_xy(root: Path) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X_train = pd.read_csv(root / X_TRAIN_PATH)
    y_train = pd.read_csv(root / Y_TRAIN_PATH).iloc[:, 0]
    X_val = pd.read_csv(root / X_VAL_PATH)
    y_val = pd.read_csv(root / Y_VAL_PATH).iloc[:, 0]
    return X_train, y_train, X_val, y_val

def infer_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "string", "bool"]).columns.tolist()

    # also treat low-cardinality ints as numeric
    for id_col in ["customer_id", "id", "customerID"]:
        if id_col in num_cols: num_cols.remove(id_col)
        if id_col in cat_cols: cat_cols.remove(id_col)

    return num_cols, cat_cols

def build_preprocess(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )

def choose_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                     objective: str = "f1",
                     min_precision: float = 0.25) -> float:
    # Evaluate many candidate thresholds
    thresholds = np.linspace(0.05, 0.95, 91)
    best_thr = 0.5
    best_score = -1.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f = f1_score(y_true, y_pred, zero_division=0)

        if objective == "f1":
            score = f
        elif objective == "recall_at_precision":
            if p < min_precision:
                continue
            score = r
        else:
            score = f

        if score > best_score:
            best_score = score
            best_thr = float(t)

    return best_thr

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    out = {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }
    return out

def save_val_predictions(met_dir: Path, model_name: str,
                         X_val: pd.DataFrame, y_val: pd.Series,
                         y_prob: np.ndarray, threshold: float) -> None:
    out = X_val.copy()
    out["y_true"] = y_val.values
    out["y_prob"] = y_prob
    out["y_pred"] = (y_prob >= threshold).astype(int)
    out.to_csv(met_dir / f"{model_name}_val_predictions.csv", index=False)

def save_plots(fig_dir: Path, model_name: str,
               y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> None:

    # ROC
    plt.figure(figsize=(6,5))
    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.title(f"{model_name} — ROC (val)")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{model_name}_roc.png")
    plt.close()

    # PR
    plt.figure(figsize=(6,5))
    PrecisionRecallDisplay.from_predictions(y_true, y_prob)
    plt.title(f"{model_name} — PR (val)")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{model_name}_pr.png")
    plt.close()

    # Confusion matrix at threshold
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    plt.imshow(cm)
    plt.title(f"{model_name} — Confusion (thr={threshold:.2f})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{model_name}_confusion.png")
    plt.close()

    # Calibration curve
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6,5))
    plt.plot(mean_pred, frac_pos, marker="o", label="Model")
    plt.plot([0,1], [0,1], linestyle="--", label="Perfect")
    plt.title(f"{model_name} — Calibration (val)")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / f"{model_name}_calibration.png")
    plt.close()


# Tuning definitions
def build_rf(preprocess: ColumnTransformer) -> Tuple[Pipeline, Dict[str, Any]]:
    clf = RandomForestClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    pipe = Pipeline([("preprocess", preprocess), ("clf", clf)])
    param_dist = {
        "clf__n_estimators": [300, 500, 800, 1200],
        "clf__max_depth": [None, 6, 10, 14],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__max_features": ["sqrt", "log2", 0.5],
    }
    return pipe, param_dist

def build_xgb(preprocess: ColumnTransformer) -> Tuple[Pipeline, Dict[str, Any]]:
    if not HAS_XGB:
        raise ImportError("xgboost not installed")

    clf = xgb.XGBClassifier(
        random_state=RANDOM_STATE,
        n_estimators=500,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        max_depth=5,
        reg_lambda=1.0,
        min_child_weight=1,
        eval_metric="logloss",
        n_jobs=-1,
    )
    pipe = Pipeline([("preprocess", preprocess), ("clf", clf)])
    param_dist = {
        "clf__n_estimators": [300, 500, 800],
        "clf__max_depth": [3, 4, 5, 6],
        "clf__learning_rate": [0.03, 0.05, 0.08, 0.1],
        "clf__subsample": [0.7, 0.85, 1.0],
        "clf__colsample_bytree": [0.7, 0.85, 1.0],
        "clf__reg_lambda": [0.5, 1.0, 2.0],
        "clf__min_child_weight": [1, 3, 5],
    }
    return pipe, param_dist

def build_ebm(preprocess: ColumnTransformer) -> Tuple[Pipeline, Dict[str, Any]]:
    if not HAS_EBM:
        raise ImportError("interpret not installed (pip install interpret)")

    # EBM is sensitive to preprocessing; one-hot is okay; keep it consistent.
    clf = ExplainableBoostingClassifier(
        random_state=RANDOM_STATE,
    )
    pipe = Pipeline([("preprocess", preprocess), ("clf", clf)])
    param_dist = {
        "clf__learning_rate": [0.01, 0.03, 0.05],
        "clf__max_bins": [64, 128, 256],
        "clf__max_interaction_bins": [16, 32],
        "clf__interactions": [0, 5, 10],
        "clf__min_samples_leaf": [2, 5, 10],
    }
    return pipe, param_dist


# Main
def tune_one(model_name: str,
             base_pipe: Pipeline,
             param_dist: Dict[str, Any],
             X_train: pd.DataFrame, y_train: pd.Series,
             X_val: pd.DataFrame, y_val: pd.Series,
             run: Dict[str, Path]) -> Dict[str, Any]:

    met_dir = run["met_dir"]
    fig_dir = run["fig_dir"]
    mod_dir = run["mod_dir"]

    search = RandomizedSearchCV(
        estimator=base_pipe,
        param_distributions=param_dist,
        n_iter=N_ITER,
        scoring="average_precision",   # aligns with imbalance
        cv=CV,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )

    search.fit(X_train, y_train)
    best = search.best_estimator_

    # Predict probabilities on validation
    y_prob = best.predict_proba(X_val)[:, 1]

    # Threshold selection
    thr = choose_threshold(
        y_true=y_val.values,
        y_prob=y_prob,
        objective=OBJECTIVE,
        min_precision=MIN_PRECISION,
    )

    # Metrics
    m = compute_metrics(y_val.values, y_prob, thr)
    m["model"] = model_name
    m["status"] = "OK"
    m["best_params"] = search.best_params_

    # Save metrics json
    (met_dir / f"{model_name}_metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")

    # Save predictions for notebook failure analysis
    save_val_predictions(met_dir, model_name, X_val, y_val, y_prob, thr)

    # Save plots
    save_plots(fig_dir, model_name, y_val.values, y_prob, thr)

    # Save model artifact
    joblib.dump(best, mod_dir / f"{model_name}.joblib")

    return m


def main():
    root = project_root_from_file()
    sys.path.insert(0, str(root))

    run = make_run_paths(root, tag="step5")
    (run["met_dir"] / "_started.txt").write_text("step5 started", encoding="utf-8")

    try:
        X_train, y_train, X_val, y_val = load_processed_xy(root)
        num_cols, cat_cols = infer_feature_types(X_train)
        preprocess = build_preprocess(num_cols, cat_cols)

        results: List[Dict[str, Any]] = []

        # RandomForest
        print("Tuning RandomForest...")
        rf_pipe, rf_params = build_rf(preprocess)
        results.append(tune_one("RandomForest_Tuned", rf_pipe, rf_params, X_train, y_train, X_val, y_val, run))

        # XGBoost
        if HAS_XGB:
            print("Tuning XGBoost...")
            xgb_pipe, xgb_params = build_xgb(preprocess)
            results.append(tune_one("XGBoost_Tuned", xgb_pipe, xgb_params, X_train, y_train, X_val, y_val, run))
        else:
            results.append({
                "model": "XGBoost_Tuned", "status": "SKIPPED",
                "reason": "xgboost not installed",
            })

        # EBM
        if HAS_EBM:
            print("Tuning EBM...")
            ebm_pipe, ebm_params = build_ebm(preprocess)
            results.append(tune_one("EBM_Tuned", ebm_pipe, ebm_params, X_train, y_train, X_val, y_val, run))
        else:
            results.append({
                "model": "EBM_Tuned", "status": "SKIPPED",
                "reason": "interpret not installed",
            })

        # Save comparison CSV
        df = pd.DataFrame(results)

        # keep best_params readable
        if "best_params" in df.columns:
            df["best_params"] = df["best_params"].apply(lambda x: json.dumps(x) if isinstance(x, dict) else x)

        df.to_csv(run["met_dir"] / "tuned_comparison_val.csv", index=False)

        # Console summary
        ok = df[df.get("status", "") == "OK"].copy()
        if len(ok) > 0:
            show_cols = [c for c in ["model","threshold","roc_auc","pr_auc","precision","recall","f1","brier"] if c in ok.columns]
            ok2 = ok.sort_values(["pr_auc","f1"], ascending=False)[show_cols]
            print("\nSummary (sorted by PR-AUC then F1):")
            print(ok2.to_string(index=False))
        else:
            print("No models completed. Check _FAILED.txt")

        (run["met_dir"] / "_finished.txt").write_text("step5 finished", encoding="utf-8")

    except Exception as e:
        (run["met_dir"] / "_FAILED.txt").write_text(
            repr(e) + "\n\n" + traceback.format_exc(),
            encoding="utf-8"
        )
        raise


if __name__ == "__main__":
    main()