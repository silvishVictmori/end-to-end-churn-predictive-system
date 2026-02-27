# src/config.py
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "customer_churn_business_dataset.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TARGET_COLUMN = "churn"
ID_COLS = ["customer_id"]

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2

# Conceptual leakage lists (keep conservative)
HIGH_LEAKAGE = [
    # if you choose to exclude them in experiments, do it via a flag
    # "csat_score", "nps_score", "survey_response",
    # "support_tickets", "avg_resolution_time", "complaint_type", "escalations"
]
MEDIUM_RISK = [
    # "last_login_days_ago", "usage_growth_rate"
]

BOUNDED_RULES = {
    "weekly_active_days": (0, 7),
    "email_open_rate": (0, 1),
    "marketing_click_rate": (0, 1),
}

@dataclass(frozen=True)
class SplitConfig:
    test_size: float = TEST_SIZE
    val_size: float = VAL_SIZE
    random_state: int = RANDOM_STATE
    stratify: bool = True