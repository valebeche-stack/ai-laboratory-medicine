# Model Card

## Model name

Synthetic Laboratory Review Classifier

## Purpose

This model is included only as an educational example of a supervised machine-learning workflow in laboratory medicine.

It demonstrates how structured laboratory variables can be used to train and evaluate a binary classifier.

## Intended use

- learning Python-based machine-learning workflows
- illustrating data preprocessing and model evaluation
- discussing performance metrics and methodological limitations
- demonstrating principles relevant to clinical decision-support research

## Not intended for

- diagnosis
- patient-level clinical decisions
- release or validation of laboratory results
- automated smear-review decisions
- replacement of expert interpretation
- deployment in a clinical laboratory

## Data

The dataset is entirely synthetic.

Predictors include:

- WBC
- hemoglobin
- platelet count
- absolute neutrophil count
- absolute lymphocyte count
- CRP
- LDH
- binary analyzer flag

The outcome variable, `review_outcome`, is artificially generated for demonstration purposes and has no validated clinical meaning.

## Model

A logistic-regression classifier is used because it is transparent, easy to interpret, and suitable for demonstrating a complete classification workflow.

## Evaluation

The script reports:

- ROC AUC
- confusion matrix
- sensitivity
- specificity
- classification report

## Key limitations

### 1. Synthetic data

The model is trained on simulated values and therefore cannot establish real-world performance.

### 2. Small sample size

The example dataset is intentionally small and is unsuitable for robust model development.

### 3. No external validation

Performance is evaluated only on a random internal train/test split.

### 4. Potential overfitting

With limited data, apparent performance may be unstable and optimistic.

### 5. No calibration assessment

The demonstration does not evaluate calibration-in-the-large, calibration slope, or calibration curves.

### 6. No clinical utility analysis

No decision-curve analysis, net benefit, workflow impact, or patient-centered outcome is evaluated.

### 7. No transportability assessment

No testing is performed across different laboratories, analyzers, populations, or prevalence settings.

## Methodological note

For real clinical-prediction research, model-development and reporting should follow contemporary prediction-model methodology and relevant reporting/risk-of-bias frameworks such as TRIPOD+AI and PROBAST+AI where applicable.

This repository does not claim compliance with those frameworks; it uses their underlying methodological principles as educational guidance.
