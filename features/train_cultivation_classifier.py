import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
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
    parser = argparse.ArgumentParser(description="Train extensif/intensif classifier from parcel Sentinel-2 features.")
    parser.add_argument("--features", default="features/parcel_sentinel2_features.csv")
    parser.add_argument("--output-dir", default="models/cultivation_classifier")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def select_feature_columns(df):
    columns = []
    for column in df.columns:
        if column in EXCLUDED_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            if df[column].notna().sum() == 0:
                continue
            columns.append(column)
    return columns


def split_xy(df, split, feature_columns):
    subset = df[df["split"] == split].copy()
    x = subset[feature_columns]
    y = subset["cultivation_system"]
    return subset, x, y


def predict_with_threshold(model, x, positive_class="intensif", threshold=None):
    if threshold is None:
        return model.predict(x)

    classes = list(model.classes_)
    if positive_class not in classes:
        raise ValueError(f"Positive class {positive_class!r} not found in model classes {classes}")

    positive_index = classes.index(positive_class)
    negative_class = [label for label in classes if label != positive_class][0]
    probabilities = model.predict_proba(x)[:, positive_index]
    return np.where(probabilities >= threshold, positive_class, negative_class)


def positive_probabilities(model, x, positive_class="intensif"):
    classes = list(model.classes_)
    positive_index = classes.index(positive_class)
    return model.predict_proba(x)[:, positive_index]


def evaluate_predictions(y, preds):
    labels = sorted(y.unique().tolist())
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "macro_f1": float(f1_score(y, preds, average="macro")),
        "weighted_f1": float(f1_score(y, preds, average="weighted")),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y, preds, labels=labels).tolist(),
        "classification_report": classification_report(y, preds, output_dict=True, zero_division=0),
    }


def evaluate(model, x, y, threshold=None):
    preds = predict_with_threshold(model, x, threshold=threshold)
    return evaluate_predictions(y, preds), preds


def tune_threshold(model, x, y, positive_class="intensif"):
    thresholds = np.linspace(0.05, 0.95, 19)
    rows = []
    for threshold in thresholds:
        preds = predict_with_threshold(model, x, positive_class=positive_class, threshold=float(threshold))
        rows.append(
            {
                "threshold": float(threshold),
                "macro_f1": float(f1_score(y, preds, average="macro")),
                "accuracy": float(accuracy_score(y, preds)),
            }
        )
    return max(rows, key=lambda row: (row["macro_f1"], row["accuracy"])), rows


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features)
    feature_columns = select_feature_columns(df)

    train_df, x_train, y_train = split_xy(df, "train", feature_columns)
    val_df, x_val, y_val = split_xy(df, "val", feature_columns)
    test_df, x_test, y_test = split_xy(df, "test", feature_columns)

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=args.n_estimators,
                    random_state=args.random_state,
                    class_weight="balanced",
                    min_samples_leaf=1,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)

    val_metrics, val_preds = evaluate(model, x_val, y_val)
    test_metrics, test_preds = evaluate(model, x_test, y_test)
    best_threshold, threshold_search = tune_threshold(model, x_val, y_val)
    val_threshold_metrics, val_threshold_preds = evaluate(model, x_val, y_val, threshold=best_threshold["threshold"])
    test_threshold_metrics, test_threshold_preds = evaluate(model, x_test, y_test, threshold=best_threshold["threshold"])

    val_prob_intensif = positive_probabilities(model, x_val)
    test_prob_intensif = positive_probabilities(model, x_test)

    predictions = pd.concat(
        [
            val_df[["id", "split", "cultivation_system", "governorate", "zone_id"]].assign(
                prediction=val_preds,
                prediction_threshold=val_threshold_preds,
                prob_intensif=val_prob_intensif,
            ),
            test_df[["id", "split", "cultivation_system", "governorate", "zone_id"]].assign(
                prediction=test_preds,
                prediction_threshold=test_threshold_preds,
                prob_intensif=test_prob_intensif,
            ),
        ],
        ignore_index=True,
    )
    predictions["correct"] = predictions["cultivation_system"] == predictions["prediction"]
    predictions["correct_threshold"] = predictions["cultivation_system"] == predictions["prediction_threshold"]
    predictions.to_csv(output_dir / "predictions.csv", index=False)

    rf = model.named_steps["rf"]
    importances = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": rf.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importances.to_csv(output_dir / "feature_importance.csv", index=False)

    summary = {
        "features_path": args.features,
        "feature_columns": feature_columns,
        "class_counts": {
            split: df[df["split"] == split]["cultivation_system"].value_counts().to_dict()
            for split in ("train", "val", "test")
        },
        "validation": val_metrics,
        "test": test_metrics,
        "threshold_tuning_on_validation": {
            "best": best_threshold,
            "all": threshold_search,
        },
        "validation_thresholded": val_threshold_metrics,
        "test_thresholded": test_threshold_metrics,
    }

    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    joblib.dump(model, output_dir / "random_forest.joblib")

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "features": len(feature_columns),
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "best_val_threshold": best_threshold["threshold"],
                "test_threshold_accuracy": test_threshold_metrics["accuracy"],
                "test_threshold_macro_f1": test_threshold_metrics["macro_f1"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
