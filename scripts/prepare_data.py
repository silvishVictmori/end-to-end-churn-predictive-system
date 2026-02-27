from datetime import datetime

from src.config import RAW_DATA_PATH, PROCESSED_DIR, TARGET_COLUMN, BOUNDED_RULES, SplitConfig
from src.data import load_raw, basic_validation_report, prepare_data_step3, save_processed_splits


def main():
    df_raw = load_raw(RAW_DATA_PATH)

    raw_report = basic_validation_report(df_raw, TARGET_COLUMN, BOUNDED_RULES)

    bundle = prepare_data_step3(
        df_raw=df_raw,
        drop_leakage=False,   # start with assumption "available at prediction time"
        cfg=SplitConfig()
    )

    split_report = {
        "train_rate": float(bundle["y_train"].mean()),
        "val_rate": float(bundle["y_val"].mean()),
        "test_rate": float(bundle["y_test"].mean()),
        "train_n": int(len(bundle["y_train"])),
        "val_n": int(len(bundle["y_val"])),
        "test_n": int(len(bundle["y_test"])),
    }

    metadata = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "raw_path": str(RAW_DATA_PATH),
        "processed_dir": str(PROCESSED_DIR),
        "target": TARGET_COLUMN,
        "raw_validation_summary": raw_report,
        "split_summary": split_report,
        "dropped_cols": bundle["dropped_cols"],
        "numeric_cols": bundle["numeric_cols"],
        "categorical_cols": bundle["categorical_cols"],
    }

    save_processed_splits(bundle, PROCESSED_DIR, metadata)

    print("✅ Saved splits to:", PROCESSED_DIR)
    print("Split churn rates:", split_report)
    if bundle["dropped_cols"]:
        print("Dropped columns:", bundle["dropped_cols"])


if __name__ == "__main__":
    main()