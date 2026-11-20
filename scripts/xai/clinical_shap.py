import os
import warnings
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from pathlib import Path

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
BASE = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE / "results" / "clinical" / "clinical_model.joblib"
TEST_PATH = BASE / "data" / "clinical" / "clinical_test.csv"

OUT_DIR = BASE / "results" / "xai" / "clinical"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Six subjects already selected for the MRI/XAI analysis
SELECTED_SUBJECTS = [
    174151,
    137842,
    3517,
    4010,
    3178,
    293727,
]

RANDOM_STATE = 42


# ---------------------------------------------------------
# LOAD MODEL AND DATA
# ---------------------------------------------------------
def load_data():
    print("Loading clinical model...")
    pipeline = joblib.load(MODEL_PATH)

    print("Loading clinical test data...")
    df = pd.read_csv(TEST_PATH)

    preprocessing = pipeline.named_steps["preprocessing"]
    model = pipeline.named_steps["model"]

    return pipeline, preprocessing, model, df


# ---------------------------------------------------------
# GET ORIGINAL MODEL FEATURES
# ---------------------------------------------------------
def get_original_features(preprocessing):
    numeric_features = []

    for name, transformer, columns in preprocessing.transformers_:
        if name == "numeric":
            numeric_features.extend(list(columns))
        elif name == "sex":
            numeric_features.extend(list(columns))

    return numeric_features


# ---------------------------------------------------------
# TRANSFORM DATA FOR RANDOM FOREST
# ---------------------------------------------------------
def transform_data(preprocessing, X):
    X_transformed = preprocessing.transform(X)

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    feature_names = list(preprocessing.get_feature_names_out())

    return np.asarray(X_transformed), feature_names


# ---------------------------------------------------------
# CONVERT SHAP VALUES TO BINARY CLASS-1 ARRAY
# ---------------------------------------------------------
def get_class1_shap_values(shap_values):
    """
    SHAP versions may return either:
      (samples, features)
    or:
      (samples, features, classes)

    This function consistently returns class-1 SHAP values.
    """
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            return np.asarray(shap_values[1])
        return np.asarray(shap_values[-1])

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        return shap_values[:, :, 1]

    return shap_values


# ---------------------------------------------------------
# GET CLASS-1 BASE VALUE
# ---------------------------------------------------------
def get_class1_base_value(explainer):
    expected = explainer.expected_value
    expected = np.asarray(expected)

    if expected.ndim == 0:
        return float(expected)

    if expected.size == 2:
        return float(expected[1])

    return float(expected.flat[-1])


# ---------------------------------------------------------
# MAP TRANSFORMED FEATURES BACK TO ORIGINAL FEATURES
# ---------------------------------------------------------
def original_feature_name(name):
    """
    Examples:
      numeric__Age       -> Age
      numeric__NP3FTAPR  -> NP3FTAPR
      sex__Sex_F         -> Sex
      sex__Sex_M         -> Sex
    """
    if "__" in name:
        name = name.split("__", 1)[1]

    if name.startswith("Sex_"):
        return "Sex"

    return name


# ---------------------------------------------------------
# GROUP SHAP VALUES
# ---------------------------------------------------------
def group_shap_values(shap_values, transformed_names):
    grouped_names = [
        original_feature_name(name)
        for name in transformed_names
    ]

    unique_names = list(dict.fromkeys(grouped_names))

    grouped = np.zeros(
        (shap_values.shape[0], len(unique_names)),
        dtype=float
    )

    for i, feature in enumerate(unique_names):
        indices = [
            j for j, name in enumerate(grouped_names)
            if name == feature
        ]

        grouped[:, i] = shap_values[:, indices].sum(axis=1)

    return grouped, unique_names


# ---------------------------------------------------------
# SAVE GLOBAL SHAP VALUES
# ---------------------------------------------------------
def save_shap_csv(df, grouped_shap, grouped_names):
    rows = []

    for i in range(len(df)):
        row = {
            "Subject": df.iloc[i]["Subject"],
            "Label": df.iloc[i]["Label"],
        }

        for j, feature in enumerate(grouped_names):
            row[f"SHAP_{feature}"] = grouped_shap[i, j]

        rows.append(row)

    output = pd.DataFrame(rows)

    output.to_csv(
        OUT_DIR / "clinical_shap_values.csv",
        index=False
    )

    print(
        f"Saved: {OUT_DIR / 'clinical_shap_values.csv'}"
    )


# ---------------------------------------------------------
# GLOBAL BAR PLOT
# ---------------------------------------------------------
def create_bar_plot(grouped_shap, grouped_names):
    importance = np.mean(
        np.abs(grouped_shap),
        axis=0
    )

    order = np.argsort(importance)[::-1]

    plt.figure(figsize=(10, 8))

    plt.barh(
        np.arange(len(order)),
        importance[order]
    )

    plt.yticks(
        np.arange(len(order)),
        [grouped_names[i] for i in order]
    )

    plt.gca().invert_yaxis()

    plt.xlabel("Mean |SHAP value|")
    plt.ylabel("Clinical feature")
    plt.title("Clinical SHAP Feature Importance")

    plt.tight_layout()

    plt.savefig(
        OUT_DIR / "clinical_shap_bar.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {OUT_DIR / 'clinical_shap_bar.png'}"
    )


# ---------------------------------------------------------
# BEESWARM PLOT
# ---------------------------------------------------------
def create_beeswarm(
    grouped_shap,
    grouped_names,
    X_original
):
    shap_explanation = shap.Explanation(
        values=grouped_shap,
        data=X_original.values,
        feature_names=grouped_names
    )

    plt.figure(figsize=(11, 9))

    shap.plots.beeswarm(
        shap_explanation,
        max_display=len(grouped_names),
        show=False
    )

    plt.title("Clinical SHAP Beeswarm")

    plt.tight_layout()

    plt.savefig(
        OUT_DIR / "clinical_shap_beeswarm.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {OUT_DIR / 'clinical_shap_beeswarm.png'}"
    )


# ---------------------------------------------------------
# GROUPED IMPORTANCE TABLE/PLOT
# ---------------------------------------------------------
def create_grouped_importance(grouped_shap, grouped_names):
    importance = np.mean(
        np.abs(grouped_shap),
        axis=0
    )

    percentage = (
        importance / importance.sum() * 100
    )

    result = pd.DataFrame({
        "Feature": grouped_names,
        "Mean_Absolute_SHAP": importance,
        "Contribution_Percent": percentage
    })

    result = result.sort_values(
        "Mean_Absolute_SHAP",
        ascending=False
    )

    result.to_csv(
        OUT_DIR / "clinical_shap_feature_importance.csv",
        index=False
    )

    print(
        f"Saved: {OUT_DIR / 'clinical_shap_feature_importance.csv'}"
    )


# ---------------------------------------------------------
# PATIENT WATERFALLS
# ---------------------------------------------------------
def create_patient_waterfalls(
    df,
    grouped_shap,
    grouped_names,
    base_value
):
    for subject in SELECTED_SUBJECTS:

        matches = np.where(
            df["Subject"].astype(str).values
            == str(subject)
        )[0]

        if len(matches) == 0:
            print(
                f"Subject {subject} not found in clinical test set."
            )
            continue

        idx = matches[0]

        explanation = shap.Explanation(
            values=grouped_shap[idx],
            base_values=base_value,
            data=df.iloc[idx][grouped_names].values,
            feature_names=grouped_names
        )

        plt.figure(figsize=(10, 8))

        shap.plots.waterfall(
            explanation,
            max_display=15,
            show=False
        )

        plt.title(
            f"Clinical SHAP Explanation - Subject {subject}"
        )

        plt.tight_layout()

        output_path = (
            OUT_DIR /
            f"clinical_shap_waterfall_{subject}.png"
        )

        plt.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()

        print(f"Saved: {output_path}")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():

    print("=" * 60)
    print("NEUROINSIGHT - CLINICAL SHAP ANALYSIS")
    print("=" * 60)

    pipeline, preprocessing, model, df = load_data()

    print("\nClinical test subjects:", len(df))
    print("Clinical model:", type(model).__name__)

    # Features used by the saved model
    feature_names = get_original_features(preprocessing)

    print(
        "Original clinical features:",
        len(feature_names)
    )

    # Only model features
    X = df[feature_names].copy()

    print("\nTransforming clinical test data...")

    X_transformed, transformed_names = transform_data(
        preprocessing,
        X
    )

    print(
        "Transformed feature count:",
        len(transformed_names)
    )

    # -----------------------------------------------------
    # SHAP
    # -----------------------------------------------------
    print("\nCreating TreeExplainer...")

    explainer = shap.TreeExplainer(model)

    print("Calculating SHAP values...")

    raw_shap_values = explainer.shap_values(
        X_transformed
    )

    shap_values = get_class1_shap_values(
        raw_shap_values
    )

    base_value = get_class1_base_value(
        explainer
    )

    print(
        "SHAP matrix shape:",
        shap_values.shape
    )

    # -----------------------------------------------------
    # GROUP TRANSFORMED FEATURES
    # -----------------------------------------------------
    grouped_shap, grouped_names = group_shap_values(
        shap_values,
        transformed_names
    )

    print(
        "Original feature SHAP shape:",
        grouped_shap.shape
    )

    # -----------------------------------------------------
    # OUTPUTS
    # -----------------------------------------------------
    save_shap_csv(
        df,
        grouped_shap,
        grouped_names
    )

    create_bar_plot(
        grouped_shap,
        grouped_names
    )

    create_beeswarm(
        grouped_shap,
        grouped_names,
        X
    )

    create_grouped_importance(
        grouped_shap,
        grouped_names
    )

    create_patient_waterfalls(
        df,
        grouped_shap,
        grouped_names,
        base_value
    )

    print("\n" + "=" * 60)
    print("CLINICAL SHAP ANALYSIS COMPLETE")
    print("=" * 60)

    print("\nOutputs:")
    print(" - clinical_shap_values.csv")
    print(" - clinical_shap_bar.png")
    print(" - clinical_shap_beeswarm.png")
    print(" - clinical_shap_feature_importance.csv")
    print(" - clinical_shap_waterfall_<subject>.png")


if __name__ == "__main__":
    main()