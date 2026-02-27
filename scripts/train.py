from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.utils_paths import make_run_paths
from src.preprocess import build_preprocess
from src.models import get_model_specs
from src.evaluate import evaluate_and_save


def main():
    project_root = Path(__file__).resolve().parents[1]
    run = make_run_paths(project_root, tag="step4")

    processed = project_root / "data" / "processed"
    if not processed.exists():
        raise FileNotFoundError(f"Processed data folder not found: {processed}")

    # 1) Load data splits

    X_train = pd.read_csv(processed / "X_train.csv")
    X_val   = pd.read_csv(processed / "X_val.csv")
    y_train = pd.read_csv(processed / "y_train.csv").iloc[:, 0]
    y_val   = pd.read_csv(processed / "y_val.csv").iloc[:, 0]

    print("Loaded splits:")
    print("  X_train:", X_train.shape, " y_train churn:", float(y_train.mean()))
    print("  X_val  :", X_val.shape,   " y_val churn:", float(y_val.mean()))
    print()


    # 2) Preprocess pipeline

    preprocess, num_cols, cat_cols = build_preprocess(X_train, scale_numeric=True)
    print(f"Preprocess: {len(num_cols)} numeric, {len(cat_cols)} categorical")
    print()


    # 3) Model specs 

    specs = get_model_specs(preprocess, y_train=y_train, random_state=42)

    all_metrics: list[dict] = []
    skipped_rows: list[dict] = []

    def write_comparison_csv():
        """Write a combined comparison file even if some models skip/fail."""
        df_ok = pd.DataFrame(all_metrics) if all_metrics else pd.DataFrame()
        df_skip = pd.DataFrame(skipped_rows) if skipped_rows else pd.DataFrame()

        wanted_cols = [
            "model", "status",
            "roc_auc", "pr_auc", "precision", "recall", "f1",
            "tp", "fp", "tn", "fn", "brier",
            "threshold", "reason"
        ]

        df_all = pd.concat([df_ok, df_skip], ignore_index=True, sort=False)
        for c in wanted_cols:
            if c not in df_all.columns:
                df_all[c] = None
        df_all = df_all[wanted_cols]

        out_path = run.met_dir / "model_comparison_val.csv"
        df_all.to_csv(out_path, index=False)


    # 4) Train + evaluate each model

    for spec in specs:
        if not spec.available:
            msg = spec.reason_unavailable or "Unavailable"
            print(f"Skipping {spec.name}: {msg}")
            skipped_rows.append({
                "model": spec.name,
                "status": "SKIPPED",
                "reason": msg
            })
            write_comparison_csv()
            continue

        try:
            print(f"Training {spec.name} ...")
            model = spec.builder()

            # Fit
            model.fit(X_train, y_train)

            # Evaluate on validation and save plots+metrics
            m = evaluate_and_save(
                model=model,
                X=X_val,
                y=y_val,
                model_name=spec.name,
                fig_dir=run.fig_dir,
                met_dir=run.met_dir,
                threshold=0.5
            )
            m["status"] = "OK"
            all_metrics.append(m)

            print(f"  Done. ROC-AUC={m['roc_auc']:.4f} PR-AUC={m['pr_auc']:.4f} "
                  f"Recall={m['recall']:.4f} F1={m['f1']:.4f}")

        except Exception as e:
            print(f"⚠️ {spec.name} FAILED:", repr(e))
            skipped_rows.append({
                "model": spec.name,
                "status": "FAILED",
                "reason": repr(e)
            })

        # Always write comparison after each model so file always exists
        write_comparison_csv()
        print()


    # 5) Final print summary
   
    out_csv = run.met_dir / "model_comparison_val.csv"
    print("Run:", run.run_id)
    print("Figures saved to:", run.fig_dir)
    print("Metrics saved to:", run.met_dir)
    print("Comparison CSV:", out_csv)

    # Show quick table
    try:
        df = pd.read_csv(out_csv)
        print("\nSummary (sorted by ROC-AUC then Recall):")
        df_ok = df[df["status"] == "OK"].copy()
        if len(df_ok):
            df_ok = df_ok.sort_values(by=["roc_auc", "recall"], ascending=False)
        print(df[["model", "status", "roc_auc", "pr_auc", "recall", "f1", "reason"]])
    except Exception as e:
        print("Could not read comparison CSV:", repr(e))


if __name__ == "__main__":
    main()