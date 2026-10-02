"""
Educational AI workflow for laboratory medicine.

All data are synthetic.
This script demonstrates a basic binary classification pipeline.
It is NOT a clinical decision-support system.
"""

import pandas as pd
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


def load_data(path="synthetic_laboratory_ai_data.csv"):
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
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    return model


def evaluate_model(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
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


def main():
    df = load_data()

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    evaluate_model(y_test, y_pred, y_prob)

    print("\nImportant:")
    print(
        "This is a synthetic educational example. "
        "Observed performance must not be interpreted as clinical validity."
    )


if __name__ == "__main__":
    main()
