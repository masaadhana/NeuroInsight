# Clinical Parkinson's Disease Model

## Final model

**Random Forest** was selected using validation ROC-AUC (then F1 and accuracy as deterministic tie-breakers). The saved complete pipeline is [`clinical_model.joblib`](clinical_model.joblib).

## Data and integrity

- Development data: `data/clinical/clinical_train.csv` (221 subjects)
- Final evaluation data: `data/clinical/clinical_test.csv` (56 subjects)
- Reference-only file: `data/clinical/clinical_v2_canonical.csv` (used only to check Subject coverage; never used for fitting, validation, tuning, or evaluation)
- Train/test Subject overlap: 0
- Duplicate Subject IDs: train = 0, test = 0
- Target distribution: train = {'0': 111, '1': 110}; test = {'0': 28, '1': 28}
- Missing clinical feature values: train = {'NP3FTAPL': 1, 'NP3HMOVL': 1, 'NP3PTRML': 1}; test = none

## Features

The model uses only Age, Sex, and the individual NP3 examination variables:

- `Age`
- `NP3SPCH`
- `NP3FACXP`
- `NP3RIGN`
- `NP3RIGRU`
- `NP3RIGLU`
- `NP3RIGRL`
- `NP3RIGLL`
- `NP3FTAPR`
- `NP3FTAPL`
- `NP3HMOVR`
- `NP3HMOVL`
- `NP3PRSPR`
- `NP3PRSPL`
- `NP3TTAPR`
- `NP3TTAPL`
- `NP3LGAGR`
- `NP3LGAGL`
- `NP3RISNG`
- `NP3GAIT`
- `NP3FRZGT`
- `NP3PSTBL`
- `NP3POSTR`
- `NP3BRADY`
- `NP3PTRMR`
- `NP3PTRML`
- `NP3KTRMR`
- `NP3KTRML`
- `NP3RTARU`
- `NP3RTALU`
- `NP3RTARL`
- `NP3RTALL`
- `NP3RTALJ`
- `NP3RTCON`
- `Sex`

Excluded from the model are `Clinical_EVENT_ID`, `Clinical_EXAMDT`, `Clinical_REC_ID`, `Group`, `Image Data ID`, `Label`, `MRI_Acq_Date`, `MRI_File`, `MRI_Visit`, `NP3TOT`, `Subject`, plus any other identifier, date, or target-derived field. In particular, `NP3TOT` is excluded because it aggregates the individual NP3 items.

## Validation strategy and preprocessing

An 80/20 stratified internal validation split of the 221-row training file was made with `random_state=42`. Logistic Regression, Random Forest, and Gradient Boosting were compared only on that split. After selection, the chosen complete preprocessing-and-model pipeline was refit on all 221 training rows. The test file was used once, only for the final evaluation.

Numeric values are median-imputed; Sex is most-frequent-imputed and one-hot encoded. Logistic Regression additionally standardizes numeric features. All preprocessing is inside the saved sklearn pipeline and is fit only on the relevant training partition.

## Metrics

| Evaluation | Accuracy | Precision | Recall | F1 | ROC-AUC | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Internal validation | 0.9778 | 0.9565 | 1.0000 | 0.9778 | 1.0000 | 22 | 1 | 0 | 22 |
| Final test | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 28 | 0 | 0 | 28 |

## Outputs

- [`clinical_metrics.csv`](clinical_metrics.csv): candidate validation and final selected-model metrics.
- [`clinical_predictions.csv`](clinical_predictions.csv): exact source `Subject` values, labels, split, class-1 probability, and predicted label for all 277 subjects.
- [`clinical_model.joblib`](clinical_model.joblib): complete preprocessing and final model pipeline.
- [`clinical_roc_curve.png`](clinical_roc_curve.png): final test ROC curve.
