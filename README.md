# AI in Laboratory Medicine

Educational examples and methodological notes on artificial intelligence applications in laboratory medicine and clinical decision support.

This repository is a small scientific portfolio project showing how a simple machine-learning workflow can be structured in a laboratory-medicine context while keeping model output separate from clinical interpretation.

## What this project demonstrates

- loading and preprocessing structured laboratory data
- splitting data into training and test sets
- fitting a simple classification model
- evaluating discrimination with ROC AUC
- examining sensitivity, specificity, and confusion matrices
- illustrating why apparent model performance is not equivalent to clinical utility
- documenting methodological limitations, bias, overfitting, and the need for external validation

## Files

- `synthetic_laboratory_ai_data.csv` — fully synthetic dataset with no patient information
- `ai_laboratory_demo.py` — simple Python classification workflow
- `MODEL_CARD.md` — transparent description of the model, intended use, assumptions, and limitations
- `requirements.txt` — Python dependencies
- `verify_reproducibility.py` — compares two independent demo executions

## Run the demo

Reference environment: **Python 3.12.14** and the dependency versions pinned in
`requirements.txt`. From a local checkout, create an isolated environment:

```bash
git clone https://github.com/valebeche-stack/ai-laboratory-medicine.git
cd ai-laboratory-medicine
python3.12 -m venv .venv
```

On Linux/macOS:

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python ai_laboratory_demo.py
```

On Windows (PowerShell; use `py -3.12 -m venv .venv` to create the environment):

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe ai_laboratory_demo.py
```

The script prints ROC AUC, sensitivity, specificity, the confusion matrix and
the classification report. It requires no graphical interface, network access
after installation, or user input. The bundled CSV is resolved relative to the
script, so execution does not depend on the current working directory. The
dataset is fixed; no new synthetic observations are generated at runtime.

For an auditable record, including unrounded probabilities and coefficients:

```bash
.venv/bin/python ai_laboratory_demo.py --output outputs/run.json
```

On Windows, replace `.venv/bin/python` with `.venv\Scripts\python.exe`.
Output paths are relative to the current working directory. An existing output
file is overwritten; use a new filename to retain an earlier run.

## Verify repeatability

```bash
.venv/bin/python verify_reproducibility.py
```

The verifier starts two fresh Python processes using the same interpreter and
runs them from a temporary directory. It checks **exact equality** of the dataset
SHA-256, ordered training/test sample IDs, observed labels, predicted labels,
full-precision probabilities, fitted coefficients and intercept, iteration count,
metrics, model configuration and environment metadata. It prints `PASS` and exits
with code 0 when they agree; a differing record or failed execution gives a
nonzero exit code. Temporary records are removed after the check. Merely comparing
the rounded console metrics would not establish this level of repeatability.

The split and classifier share `RANDOM_STATE = 42`; the split remains stratified
with 30% held out. The solver is explicitly `lbfgs`, with `C=1.0`, `tol=1e-4`
and `max_iter=1000`. **`lbfgs` does not use `random_state`**; the classifier seed
documents the intended configuration but is not evidence that the original
default solver was stochastic. See the
[scikit-learn parameter documentation](https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.LogisticRegression.html).
Numerical-library threads are limited to one during fitting and prediction.
A convergence warning causes execution to fail rather than reporting an
unconverged model as a successful run. Preprocessing is still fitted only on
the training set through the pipeline.

This procedure establishes repeatability **within the same environment, with
the same dataset bytes and row order**. Pinned dependencies and recorded Python,
platform and package versions aid reproduction; they do not guarantee identical
floating-point results across operating systems, CPU architectures, numerical
libraries or Python versions. Such comparisons require a separate tolerance-based
assessment. Use the same repository commit when reproducing a saved run.

Repeatability is not a measure of model validity, uncertainty, calibration,
generalizability or clinical utility. Sensitivity and specificity describe the
synthetic held-out set at the classifier's default decision threshold; this is
not a clinically selected or validated operating threshold.

## Scientific perspective

AI models in laboratory medicine should not be judged only by headline accuracy or ROC AUC. Evaluation should consider:

- participant and data-source selection
- representativeness of the study population
- predictor definition and measurement
- outcome definition
- missing-data handling
- sample size and event rate
- risk of overfitting
- internal versus external validation
- calibration
- clinically meaningful operating thresholds
- transportability across instruments, laboratories, and populations
- potential workflow and automation bias

A model can perform well statistically and still fail to provide clinically useful or generalizable decision support.

## Demo task

The example model predicts a **synthetic review outcome** from a small set of laboratory variables. This is a programming demonstration only.

The target does **not** represent a validated diagnosis, clinical endpoint, or approved laboratory decision rule.

## Author

**Valentina Becherucci**  
Clinical Laboratory Biologist  
Laboratory Medicine | Clinical Pathology | Hematology | Biomedical Research | AI in Laboratory Medicine

Google Scholar: https://scholar.google.com/citations?user=14LEcucAAAAJ&hl=en

## Disclaimer

All data are synthetic. No patient data are included. This repository is for educational and portfolio purposes only and must not be used for clinical decision-making or patient care.
