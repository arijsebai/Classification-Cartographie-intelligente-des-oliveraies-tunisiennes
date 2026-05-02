import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline


EXCLUDED_COLUMNS = {
    "id",
    "image_path",
    "split",
    "cultivation_system",
    "governorate",
    "zone_id",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Spatial CV classifier for extensif/intensif/hyper-intensif parcels.")
    parser.add_argument("--features", default="features/parcel_sentinel2_features.csv")
    parser.add_argument("--output-dir", default="models/spatial_cv_classifier")
    parser.add_argument("--model", choices=["rf", "xgboost"], default="rf")
    parser.add_argument("--group-column", choices=["zone_id", "governorate"], default="zone_id")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def select_feature_columns(df):
    columns = []
    for column in df.columns:
        if column in EXCLUDED_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[column]) and df[column].notna().sum() > 0:
            columns.append(column)
    return columns


def make_classifier(args):
    if args.model == "rf":
        estimator = RandomForestClassifier(
            n_estimators=args.n_estimators,
            random_state=args.random_state,
            class_weight="balanced",
            min_samples_leaf=1,
        )
    else:
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("Install xgboost or use --model rf") from exc

        estimator = XGBClassifier(
            n_estimators=args.n_estimators,
            random_state=args.random_state,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
        )

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", estimator),
        ]
    )


def build_cv(groups, requested_folds):
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("Spatial CV needs at least two groups.")

    if len(unique_groups) <= requested_folds:
        return LeaveOneGroupOut(), len(unique_groups), "LeaveOneGroupOut"

    folds = min(requested_folds, len(unique_groups))
    return GroupKFold(n_splits=folds), folds, f"GroupKFold(n_splits={folds})"


def evaluate(y_true, y_pred, labels):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features)
    df = df[df["cultivation_system"].notna()].copy()
    feature_columns = select_feature_columns(df)
    labels = sorted(df["cultivation_system"].unique().tolist())

    if len(labels) < 3:
        print(
            json.dumps(
                {
                    "warning": "Dataset currently has fewer than 3 classes. Script is multi-class ready, but current labels are only those listed.",
                    "classes": labels,
                },
                indent=2,
            )
        )

    x = df[feature_columns]
    y = df["cultivation_system"].astype(str)
    groups = df[args.group_column].astype(str)
    cv, fold_count, cv_name = build_cv(groups.values, args.folds)

    fold_rows = []
    prediction_rows = []

    for fold_index, (train_idx, test_idx) in enumerate(cv.split(x, y, groups), start=1):
        model = make_classifier(args)
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(x.iloc[test_idx])

        fold_metrics = evaluate(y.iloc[test_idx], preds, labels=labels)
        fold_groups = sorted(groups.iloc[test_idx].unique().tolist())
        fold_rows.append(
            {
                "fold": fold_index,
                "groups": fold_groups,
                "train_count": int(len(train_idx)),
                "test_count": int(len(test_idx)),
                **fold_metrics,
            }
        )

        for row_index, pred in zip(test_idx, preds):
            row = df.iloc[row_index]
            prediction_rows.append(
                {
                    "fold": fold_index,
                    "id": row["id"],
                    "group": row[args.group_column],
                    "split": row.get("split"),
                    "cultivation_system": row["cultivation_system"],
                    "prediction": pred,
                    "correct": bool(row["cultivation_system"] == pred),
                }
            )

    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(output_dir / "spatial_cv_predictions.csv", index=False)

    overall_metrics = evaluate(predictions["cultivation_system"], predictions["prediction"], labels=labels)
    summary = {
        "features_path": args.features,
        "model": args.model,
        "group_column": args.group_column,
        "cv": cv_name,
        "fold_count": fold_count,
        "classes": labels,
        "class_counts": df["cultivation_system"].value_counts().to_dict(),
        "group_counts": df[args.group_column].value_counts().to_dict(),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "overall": overall_metrics,
        "folds": fold_rows,
    }
    (output_dir / "spatial_cv_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    final_model = make_classifier(args)
    final_model.fit(x, y)
    joblib.dump(final_model, output_dir / f"{args.model}_final.joblib")

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "model": args.model,
                "group_column": args.group_column,
                "cv": cv_name,
                "classes": labels,
                "feature_count": len(feature_columns),
                "spatial_cv_accuracy": overall_metrics["accuracy"],
                "spatial_cv_macro_f1": overall_metrics["macro_f1"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
