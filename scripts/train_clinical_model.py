"""Train and evaluate the leakage-safe clinical Parkinson's classifier.

Only ``clinical_train.csv`` is used to select the model.  The official test
file is loaded after selection and evaluated exactly once by this script.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_SEED = 42
VALIDATION_FRACTION = 0.20
ROOT = Path(__file__).resolve().parents[1]
CLINICAL_DATA = ROOT / "data" / "clinical"
RESULTS = ROOT / "results" / "clinical"

NUMERIC_FEATURES = [
    "Age",
    "NP3SPCH", "NP3FACXP", "NP3RIGN", "NP3RIGRU", "NP3RIGLU", "NP3RIGRL",
    "NP3RIGLL", "NP3FTAPR", "NP3FTAPL", "NP3HMOVR", "NP3HMOVL", "NP3PRSPR",
    "NP3PRSPL", "NP3TTAPR", "NP3TTAPL", "NP3LGAGR", "NP3LGAGL", "NP3RISNG",
    "NP3GAIT", "NP3FRZGT", "NP3PSTBL", "NP3POSTR", "NP3BRADY", "NP3PTRMR",
    "NP3PTRML", "NP3KTRMR", "NP3KTRML", "NP3RTARU", "NP3RTALU", "NP3RTARL",
    "NP3RTALL", "NP3RTALJ", "NP3RTCON",
]
CATEGORICAL_FEATURES = ["Sex"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
FORBIDDEN_FEATURES = {
    "Label", "Group", "Subject", "MRI_File", "Image Data ID", "MRI_Visit",
    "MRI_Acq_Date", "Clinical_EVENT_ID", "Clinical_EXAMDT", "Clinical_REC_ID", "NP3TOT",
}


def validate_training_inputs(train: pd.DataFrame) -> dict:
    """Validate train-only inputs before validation split and model selection."""
    if len(train) != 221:
        raise ValueError(f"Expected 221 train rows; found {len(train)}.")
    if train["Subject"].duplicated().any():
        raise ValueError("Duplicate Subject IDs in train split.")
    missing_features = set(FEATURES) - set(train.columns)
    if missing_features:
        raise ValueError(f"train is missing required features: {sorted(missing_features)}")

    feature_set = set(FEATURES)
    if feature_set & FORBIDDEN_FEATURES:
        raise ValueError("A forbidden column entered the feature matrix.")
    if set(FEATURES) != {"Age", "Sex", *NUMERIC_FEATURES[1:]}:
        raise ValueError("Feature list does not match the approved clinical feature set.")

    return {
        "duplicate_subjects_train": int(train["Subject"].duplicated().sum()),
        "train_target_distribution": {str(k): int(v) for k, v in train["Label"].value_counts().sort_index().items()},
        "train_missing_values_by_feature": {
            key: int(value) for key, value in train[FEATURES].isna().sum().items() if value
        },
    }


def validate_final_evaluation_inputs(
    train: pd.DataFrame, test: pd.DataFrame, canonical: pd.DataFrame, checks: dict
) -> dict:
    """Validate the held-out test and reference data after model selection."""
    if len(test) != 56:
        raise ValueError(f"Expected 56 test rows; found {len(test)}.")
    if test["Subject"].duplicated().any():
        raise ValueError("Duplicate Subject IDs in test split.")
    missing_features = set(FEATURES) - set(test.columns)
    if missing_features:
        raise ValueError(f"test is missing required features: {sorted(missing_features)}")

    train_subjects = set(train["Subject"])
    test_subjects = set(test["Subject"])
    overlap = train_subjects & test_subjects
    if overlap:
        raise ValueError(f"Train/test Subject overlap detected: {sorted(overlap)[:5]}")

    canonical_subjects = set(canonical["Subject"])
    all_source_subjects = train_subjects | test_subjects
    if not all_source_subjects.issubset(canonical_subjects):
        raise ValueError("Canonical dataset does not contain every train/test Subject.")
    if len(canonical) != 277 or canonical["Subject"].duplicated().any():
        raise ValueError("Canonical dataset integrity check failed.")

    checks.update({
        "train_test_subject_overlap": len(overlap),
        "canonical_contains_all_train_test_subjects": True,
        "duplicate_subjects_test": int(test["Subject"].duplicated().sum()),
        "test_target_distribution": {str(k): int(v) for k, v in test["Label"].value_counts().sort_index().items()},
        "test_missing_values_by_feature": {
            key: int(value) for key, value in test[FEATURES].isna().sum().items() if value
        },
    })
    return checks


def make_pipeline(model_name: str) -> Pipeline:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if model_name == "Logistic Regression":
        numeric_steps.append(("scaler", StandardScaler()))

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
            (
                "sex",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, random_state=RANDOM_SEED, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_SEED),
    }
    return Pipeline([("preprocessing", preprocessor), ("model", models[model_name])])


def metric_values(y_true: pd.Series, probabilities: np.ndarray) -> dict:
    predicted = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, predicted),
        "precision": precision_score(y_true, predicted, zero_division=0),
        "recall": recall_score(y_true, predicted, zero_division=0),
        "f1": f1_score(y_true, predicted, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def prefixed_metrics(prefix: str, values: dict | None) -> dict:
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "tn", "fp", "fn", "tp"]
    return {f"{prefix}_{key}": (values[key] if values else np.nan) for key in keys}


def write_readme(final_model: str, validation: dict, test: dict, checks: dict) -> None:
    feature_lines = "\n".join(f"- `{feature}`" for feature in FEATURES)
    excluded = ", ".join(f"`{feature}`" for feature in sorted(FORBIDDEN_FEATURES))
    text = f"""# Clinical Parkinson's Disease Model

## Final model

**{final_model}** was selected using validation ROC-AUC (then F1 and accuracy as deterministic tie-breakers). The saved complete pipeline is [`clinical_model.joblib`](clinical_model.joblib).

## Data and integrity

- Development data: `data/clinical/clinical_train.csv` (221 subjects)
- Final evaluation data: `data/clinical/clinical_test.csv` (56 subjects)
- Reference-only file: `data/clinical/clinical_v2_canonical.csv` (used only to check Subject coverage; never used for fitting, validation, tuning, or evaluation)
- Train/test Subject overlap: {checks['train_test_subject_overlap']}
- Duplicate Subject IDs: train = {checks['duplicate_subjects_train']}, test = {checks['duplicate_subjects_test']}
- Target distribution: train = {checks['train_target_distribution']}; test = {checks['test_target_distribution']}
- Missing clinical feature values: train = {checks['train_missing_values_by_feature'] or 'none'}; test = {checks['test_missing_values_by_feature'] or 'none'}

## Features

The model uses only Age, Sex, and the individual NP3 examination variables:

{feature_lines}

Excluded from the model are {excluded}, plus any other identifier, date, or target-derived field. In particular, `NP3TOT` is excluded because it aggregates the individual NP3 items.

## Validation strategy and preprocessing

An 80/20 stratified internal validation split of the 221-row training file was made with `random_state={RANDOM_SEED}`. Logistic Regression, Random Forest, and Gradient Boosting were compared only on that split. After selection, the chosen complete preprocessing-and-model pipeline was refit on all 221 training rows. The test file was used once, only for the final evaluation.

Numeric values are median-imputed; Sex is most-frequent-imputed and one-hot encoded. Logistic Regression additionally standardizes numeric features. All preprocessing is inside the saved sklearn pipeline and is fit only on the relevant training partition.

## Metrics

| Evaluation | Accuracy | Precision | Recall | F1 | ROC-AUC | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Internal validation | {validation['accuracy']:.4f} | {validation['precision']:.4f} | {validation['recall']:.4f} | {validation['f1']:.4f} | {validation['roc_auc']:.4f} | {validation['tn']} | {validation['fp']} | {validation['fn']} | {validation['tp']} |
| Final test | {test['accuracy']:.4f} | {test['precision']:.4f} | {test['recall']:.4f} | {test['f1']:.4f} | {test['roc_auc']:.4f} | {test['tn']} | {test['fp']} | {test['fn']} | {test['tp']} |

## Outputs

- [`clinical_metrics.csv`](clinical_metrics.csv): candidate validation and final selected-model metrics.
- [`clinical_predictions.csv`](clinical_predictions.csv): exact source `Subject` values, labels, split, class-1 probability, and predicted label for all 277 subjects.
- [`clinical_model.joblib`](clinical_model.joblib): complete preprocessing and final model pipeline.
- [`clinical_roc_curve.png`](clinical_roc_curve.png): final test ROC curve.
"""
    (RESULTS / "CLINICAL_README.md").write_text(text, encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(CLINICAL_DATA / "clinical_train.csv")
    checks = validate_training_inputs(train)

    X_train, X_validation, y_train, y_validation = train_test_split(
        train[FEATURES], train["Label"], test_size=VALIDATION_FRACTION,
        stratify=train["Label"], random_state=RANDOM_SEED,
    )
    candidate_names = ["Logistic Regression", "Random Forest", "Gradient Boosting"]
    candidate_metrics: dict[str, dict] = {}
    for model_name in candidate_names:
        pipeline = make_pipeline(model_name)
        pipeline.fit(X_train, y_train)
        candidate_metrics[model_name] = metric_values(
            y_validation, pipeline.predict_proba(X_validation)[:, 1]
        )

    final_model_name = max(
        candidate_names,
        key=lambda name: (
            candidate_metrics[name]["roc_auc"], candidate_metrics[name]["f1"],
            candidate_metrics[name]["accuracy"], -candidate_names.index(name),
        ),
    )
    final_pipeline = make_pipeline(final_model_name)
    final_pipeline.fit(train[FEATURES], train["Label"])

    # The held-out test file is not loaded or inspected until model selection
    # and the all-training-data refit above are complete.
    test = pd.read_csv(CLINICAL_DATA / "clinical_test.csv")
    canonical = pd.read_csv(CLINICAL_DATA / "clinical_v2_canonical.csv", usecols=["Subject"])
    checks = validate_final_evaluation_inputs(train, test, canonical, checks)
    test_probabilities = final_pipeline.predict_proba(test[FEATURES])[:, 1]
    test_metrics = metric_values(test["Label"], test_probabilities)

    metric_rows = []
    for rank, name in enumerate(
        sorted(candidate_names, key=lambda item: (
            candidate_metrics[item]["roc_auc"], candidate_metrics[item]["f1"],
            candidate_metrics[item]["accuracy"], -candidate_names.index(item),
        ), reverse=True), start=1
    ):
        metric_rows.append({
            "stage": "candidate_internal_validation", "selection_rank": rank,
            "model_name": name, **prefixed_metrics("validation", candidate_metrics[name]),
            **prefixed_metrics("test", None),
        })
    metric_rows.append({
        "stage": "selected_model_final_test", "selection_rank": 1,
        "model_name": final_model_name,
        **prefixed_metrics("validation", candidate_metrics[final_model_name]),
        **prefixed_metrics("test", test_metrics),
    })
    pd.DataFrame(metric_rows).to_csv(RESULTS / "clinical_metrics.csv", index=False)

    train_probabilities = final_pipeline.predict_proba(train[FEATURES])[:, 1]
    predictions = pd.concat([
        pd.DataFrame({
            "Subject": train["Subject"], "Label": train["Label"], "Split": "train",
            "PD_Probability": train_probabilities,
            "Predicted_Label": (train_probabilities >= 0.5).astype(int),
        }),
        pd.DataFrame({
            "Subject": test["Subject"], "Label": test["Label"], "Split": "test",
            "PD_Probability": test_probabilities,
            "Predicted_Label": (test_probabilities >= 0.5).astype(int),
        }),
    ], ignore_index=True)
    expected_prediction_columns = ["Subject", "Label", "Split", "PD_Probability", "Predicted_Label"]
    if predictions.columns.tolist() != expected_prediction_columns or len(predictions) != 277:
        raise ValueError("Prediction output schema check failed.")
    predictions.to_csv(RESULTS / "clinical_predictions.csv", index=False)
    joblib.dump(final_pipeline, RESULTS / "clinical_model.joblib")

    fpr, tpr, _ = roc_curve(test["Label"], test_probabilities)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{final_model_name} (AUC = {auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Clinical model: final test ROC curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(RESULTS / "clinical_roc_curve.png", dpi=160)
    plt.close()

    write_readme(final_model_name, candidate_metrics[final_model_name], test_metrics, checks)
    (RESULTS / "clinical_integrity_checks.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "final_model": final_model_name,
        "validation_metrics": candidate_metrics[final_model_name],
        "test_metrics": test_metrics,
        "results_dir": str(RESULTS),
    }, indent=2))


if __name__ == "__main__":
    main()
