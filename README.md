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
