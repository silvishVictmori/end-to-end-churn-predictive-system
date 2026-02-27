from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance


def get_feature_names(pipeline) -> np.ndarray:
    pre = pipeline.named_steps["preprocess"]
    return pre.get_feature_names_out()


def tree_importance(pipeline) -> pd.DataFrame | None:
    clf = pipeline.named_steps.get("clf", None)
    if clf is None:
        return None
    if hasattr(clf, "feature_importances_"):
        names = get_feature_names(pipeline)
        imp = clf.feature_importances_
        df = pd.DataFrame({"feature": names, "importance": imp}).sort_values("importance", ascending=False)
        return df
    return None


def logistic_coefficients(pipeline) -> pd.DataFrame | None:
    clf = pipeline.named_steps.get("clf", None)
    if clf is None or not hasattr(clf, "coef_"):
        return None
    names = get_feature_names(pipeline)
    coef = clf.coef_.ravel()
    df = pd.DataFrame({"feature": names, "coef": coef, "abs_coef": np.abs(coef)}).sort_values("abs_coef", ascending=False)
    return df


def permutation_importance_df(pipeline, X: pd.DataFrame, y: pd.Series, scoring: str = "average_precision", n_repeats: int = 10, random_state: int = 42) -> pd.DataFrame:
    result = permutation_importance(
        pipeline, X, y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )
    names = get_feature_names(pipeline)
    df = pd.DataFrame({
        "feature": names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values("importance_mean", ascending=False)
    return df


def save_top_importance_bar(df: pd.DataFrame, title: str, out_path: Path, top_n: int = 20, col: str = "importance") -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    top = df.head(top_n).copy()

    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"][::-1], top[col][::-1])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path