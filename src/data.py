# src/data.py
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import TARGET_COLUMN, ID_COLS, HIGH_LEAKAGE, MEDIUM_RISK, SplitConfig


def load_raw(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def basic_validation_report(df: pd.DataFrame, target: str, bounded_rules: Dict) -> Dict:
    report = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_pct_top10": (df.isna().mean() * 100).sort_values(ascending=False).head(10).round(2).to_dict(),
    }
    if target in df.columns:
        report["target_rate"] = float(df[target].mean())
        report["target_counts"] = {str(k): int(v) for k, v in df[target].value_counts().to_dict().items()}

    violations = {}
    for col, (lo, hi) in bounded_rules.items():
        if col in df.columns:
            violations[col] = int((~df[col].between(lo, hi)).sum())
    report["bounded_rule_violations"] = violations
    return report


def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: SplitConfig = SplitConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    strat = y if cfg.stratify else None

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=strat
    )

    strat_tv = y_trainval if cfg.stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=cfg.val_size,
        random_state=cfg.random_state,
        stratify=strat_tv,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocess_pipeline(
    X: pd.DataFrame,
    scale_numeric: bool = True
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    num_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(num_steps)

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocess, num_cols, cat_cols


# Step 3 bundle:
# - safe structural cleaning: drop duplicates, drop ID columns
# - optional leakage drop (flag-controlled)
# - split train/val/test (stratified)
# - build preprocessing pipeline (fit later in Step 4)

def prepare_data_step3(
    df_raw: pd.DataFrame,
    drop_leakage: bool = False,
    extra_drop: Optional[List[str]] = None,
    cfg: SplitConfig = SplitConfig(),
) -> Dict:
    df = df_raw.copy().drop_duplicates()

    # Drop ID cols from features
    for c in ID_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    # Optional leakage dropping (do NOT do this silently)
    drop_cols = set()
    if drop_leakage:
        drop_cols.update([c for c in (HIGH_LEAKAGE + MEDIUM_RISK) if c in X.columns])
    if extra_drop:
        drop_cols.update([c for c in extra_drop if c in X.columns])
    if drop_cols:
        X = X.drop(columns=list(drop_cols))

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y, cfg=cfg)

    preprocess, num_cols, cat_cols = build_preprocess_pipeline(X_train)

    bundle = {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "preprocess": preprocess,
        "numeric_cols": num_cols,
        "categorical_cols": cat_cols,
        "dropped_cols": sorted(list(drop_cols)),
        "split_config": cfg,
    }
    return bundle


def save_processed_splits(bundle: Dict, out_dir: Path, metadata: Dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle["X_train"].to_csv(out_dir / "X_train.csv", index=False)
    bundle["X_val"].to_csv(out_dir / "X_val.csv", index=False)
    bundle["X_test"].to_csv(out_dir / "X_test.csv", index=False)

    bundle["y_train"].to_csv(out_dir / "y_train.csv", index=False, header=True)
    bundle["y_val"].to_csv(out_dir / "y_val.csv", index=False, header=True)
    bundle["y_test"].to_csv(out_dir / "y_test.csv", index=False, header=True)

    with open(out_dir / "split_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)