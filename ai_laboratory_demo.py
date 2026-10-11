"""
Educational AI workflow for laboratory medicine.

All data are synthetic.
This script demonstrates a basic binary classification pipeline.
It is NOT a clinical decision-support system.
"""

import argparse
import hashlib
import json
import platform
import warnings
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "wbc",
    "hb",
    "plt",
    "neut_abs",
    "lymph_abs",
    "crp",
    "ldh",
    "analyzer_flag",
]

TARGET = "review_outcome"
RANDOM_STATE = 42
DEFAULT_DATA = Path(__file__).resolve().parent / "synthetic_laboratory_ai_data.csv"


def load_data(path=DEFAULT_DATA):
    df = pd.read_csv(path)
    return df


def build_model():
    numeric_features = [
        "wbc",
        "hb",
        "plt",
        "neut_abs",
        "lymph_abs",
        "crp",
        "ldh",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
        ],
        remainder="passthrough",
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", LogisticRegression(
                solver="lbfgs", C=1.0, tol=1e-4,
                max_iter=1000, random_state=RANDOM_STATE,
            )),
        ]
    )
    return model


def evaluate_model(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    auc = roc_auc_score(y_true, y_prob)

    print("\nModel evaluation")
    print("----------------")
    print(f"ROC AUC: {auc:.3f}")
    print(f"Sensitivity: {sensitivity:.3f}")
    print(f"Specificity: {specificity:.3f}")

    print("\nConfusion matrix")
    print(cm)

    print("\nClassification report")
    print(classification_report(y_true, y_pred, digits=3))
    return {
        "roc_auc": float(auc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True,
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write a reproducibility JSON record")
    args = parser.parse_args()
    df = load_data()

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = build_model()
    # Limit numerical-library threads; do not silently accept an unconverged fit.
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    metrics = evaluate_model(y_test, y_pred, y_prob)
    if args.output is not None:
        classifier = model.named_steps["classifier"]
        record = {
            "dataset_sha256": hashlib.sha256(DEFAULT_DATA.read_bytes()).hexdigest(),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "packages": {name: version(name) for name in (
                    "pandas", "scikit-learn", "numpy", "scipy", "joblib",
                    "threadpoolctl", "python-dateutil", "pytz", "tzdata", "six",
                )},
            },
            "configuration": {
                "random_state": RANDOM_STATE, "test_size": 0.30,
                "stratified": True, "features": FEATURES, "target": TARGET,
                "numerical_threads": 1,
                "classifier": classifier.get_params(),
            },
            "train_sample_ids": df.loc[X_train.index, "sample_id"].tolist(),
            "test_sample_ids": df.loc[X_test.index, "sample_id"].tolist(),
            "y_true": y_test.tolist(), "y_pred": y_pred.tolist(),
            "y_probability": y_prob.tolist(),
            "classes": classifier.classes_.tolist(),
            "transformed_features": model.named_steps["preprocess"].get_feature_names_out().tolist(),
            "coefficients": classifier.coef_.tolist(),
            "intercept": classifier.intercept_.tolist(),
            "iterations": classifier.n_iter_.tolist(),
            "metrics": metrics,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    print("\nImportant:")
    print(
        "This is a synthetic educational example. "
        "Observed performance must not be interpreted as clinical validity."
    )


if __name__ == "__main__":
    main()
