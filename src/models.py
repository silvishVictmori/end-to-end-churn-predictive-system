from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


@dataclass(frozen=True)
class ModelSpec:
    """A light wrapper so train.py can stay simple."""
    name: str
    builder: Callable[[], Pipeline]
    available: bool
    reason_unavailable: Optional[str] = None


def build_logistic(preprocess, random_state: int = 42) -> ModelSpec:
    def _make():
        return Pipeline([
            ("preprocess", preprocess),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=random_state
            ))
        ])
    return ModelSpec(name="LogisticRegression", builder=_make, available=True)


def build_rf(preprocess, random_state: int = 42) -> ModelSpec:
    def _make():
        return Pipeline([
            ("preprocess", preprocess),
            ("clf", RandomForestClassifier(
                n_estimators=500,
                random_state=random_state,
                n_jobs=-1,
                class_weight="balanced"
            ))
        ])
    return ModelSpec(name="RandomForest", builder=_make, available=True)


def build_lightgbm(preprocess, random_state: int = 42) -> ModelSpec:
    try:
        from lightgbm import LGBMClassifier
    except Exception as e:
        return ModelSpec(
            name="LightGBM",
            builder=lambda: None,  # won't be called
            available=False,
            reason_unavailable=f"lightgbm not installed: {repr(e)}"
        )

    def _make():
        model = LGBMClassifier(
            n_estimators=1200,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            class_weight="balanced"
        )
        return Pipeline([("preprocess", preprocess), ("clf", model)])

    return ModelSpec(name="LightGBM", builder=_make, available=True)


def build_xgboost(preprocess, y_train, random_state: int = 42) -> ModelSpec:
    try:
        from xgboost import XGBClassifier
    except Exception as e:
        return ModelSpec(
            name="XGBoost",
            builder=lambda: None,
            available=False,
            reason_unavailable=f"xgboost not installed: {repr(e)}"
        )

    # handle class imbalance: scale_pos_weight = neg/pos
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / max(pos, 1)

    def _make():
        model = XGBClassifier(
            n_estimators=1200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight
        )
        return Pipeline([("preprocess", preprocess), ("clf", model)])

    return ModelSpec(name="XGBoost", builder=_make, available=True)


def build_ebm(preprocess, random_state: int = 42) -> ModelSpec:
    """
    Explainable Boosting Machine (EBM) is optional.
    Dependency: interpret
    """
    try:
        from interpret.glassbox import ExplainableBoostingClassifier
    except Exception as e:
        return ModelSpec(
            name="EBM",
            builder=lambda: None,
            available=False,
            reason_unavailable=f"interpret not installed: {repr(e)}"
        )

    def _make():
        model = ExplainableBoostingClassifier(random_state=random_state)
        return Pipeline([("preprocess", preprocess), ("clf", model)])

    return ModelSpec(name="EBM", builder=_make, available=True)


def get_model_specs(preprocess, y_train, random_state: int = 42) -> List[ModelSpec]:
    """
    Returns all desired models; each spec indicates whether it is available.
    Train code can simply loop and skip unavailable ones.
    """
    return [
        build_logistic(preprocess, random_state=random_state),
        build_rf(preprocess, random_state=random_state),
        build_lightgbm(preprocess, random_state=random_state),
        build_xgboost(preprocess, y_train=y_train, random_state=random_state),
        build_ebm(preprocess, random_state=random_state),
    ]