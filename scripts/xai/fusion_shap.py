import pandas as pd
import numpy as np
from pathlib import Path
import shap
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

BASE = Path(__file__).resolve().parents[2]

CLINICAL_PREDICTIONS = (
    BASE / "results" / "clinical" / "clinical_predictions.csv"
)

MRI_TEST_PREDICTIONS = (
    BASE / "models" / "mri" / "multiview_resnet50"
    / "results" / "subject_predictions.csv"
)

MRI_TRAIN_PREDICTIONS = (
    BASE / "models" / "mri" / "multiview_resnet50"
    / "results" / "train_subject_predictions.csv"
)

EXISTING_FUSION = (
    BASE / "fusion_check" / "fusion_outputs"
    / "fusion_predictions.csv"
)

OUT_DIR = BASE / "results" / "xai" / "fusion"
OUT_DIR.mkdir(parents=True, exist_ok=True)


FEATURES = [
    "MRI_PD_Probability",
    "Clinical_PD_Probability",
]


SELECTED_SUBJECTS = [
    174151,
    137842,
    3517,
    4010,
    3178,
    293727,
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    clinical = pd.read_csv(CLINICAL_PREDICTIONS)

    mri_test = pd.read_csv(MRI_TEST_PREDICTIONS)

    mri_train = pd.read_csv(MRI_TRAIN_PREDICTIONS)

    mri_all = pd.concat(
        [
            mri_train.assign(Split="train"),
            mri_test.assign(Split="test"),
        ],
        ignore_index=True,
    )

    clinical = clinical[
        ["Subject", "Label", "Split", "PD_Probability"]
    ].rename(
        columns={
            "PD_Probability": "Clinical_PD_Probability"
        }
    )

    mri_all = mri_all[
        ["Subject", "MRI_PD_Probability"]
    ]

    merged = pd.merge(
        clinical,
        mri_all,
        on="Subject",
        how="inner",
    )

    merged = merged.sort_values("Subject").reset_index(drop=True)

    return merged


# ============================================================
# TRAIN EXACT FUSION MODEL
# ============================================================

def train_fusion_model(df):

    train_df = df[df["Split"] == "train"].reset_index(drop=True)

    test_df = df[df["Split"] == "test"].reset_index(drop=True)

    X_train = train_df[FEATURES].values
    y_train = train_df["Label"].values

    X_test = test_df[FEATURES].values

    mlp = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        max_iter=3000,
        random_state=RANDOM_STATE,
    )

    mlp.fit(X_train, y_train)

    return mlp, train_df, test_df


# ============================================================
# VERIFY AGAINST EXISTING FUSION OUTPUT
# ============================================================

def verify_model(mlp, df):

    if not EXISTING_FUSION.exists():

        print("\nExisting fusion_predictions.csv not found.")
        print("Skipping prediction verification.")

        return

    existing = pd.read_csv(EXISTING_FUSION)

    current = df.copy()

    current["Fusion_PD_Probability"] = mlp.predict_proba(
        current[FEATURES].values
    )[:, 1]

    check = current.merge(
        existing[
            [
                "Subject",
                "Fusion_PD_Probability",
            ]
        ],
        on="Subject",
        suffixes=("_new", "_existing"),
    )

    difference = np.abs(
        check["Fusion_PD_Probability_new"]
        - check["Fusion_PD_Probability_existing"]
    )

    print("\n=== FUSION MODEL VERIFICATION ===")

    print(
        f"Subjects compared: {len(check)}"
    )

    print(
        f"Maximum probability difference: "
        f"{difference.max():.12f}"
    )

    print(
        f"Mean probability difference: "
        f"{difference.mean():.12f}"
    )

    if difference.max() < 1e-8:

        print(
            "VERIFICATION PASSED: "
            "Recreated MLP matches existing Fusion predictions."
        )

    else:

        print(
            "WARNING: Recreated MLP does not exactly match "
            "the existing Fusion predictions."
        )


# ============================================================
# SHAP ANALYSIS
# ============================================================

def calculate_shap(mlp, train_df, test_df):

    X_train = train_df[FEATURES]

    X_test = test_df[FEATURES]

    # Use class-1 probability as the explained output.
    def predict_pd_probability(X):

        X = np.asarray(X)

        return mlp.predict_proba(X)[:, 1]

    # Background data for SHAP.
    background = shap.sample(
        X_train,
        min(100, len(X_train)),
        random_state=RANDOM_STATE,
    )

    print("\nCreating SHAP explainer...")

    explainer = shap.KernelExplainer(
        predict_pd_probability,
        background,
    )

    print("Calculating SHAP values...")

    shap_values = explainer.shap_values(
        X_test,
        nsamples=100,
    )

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:

        shap_values = shap_values[:, :, 0]

    print(
        f"SHAP matrix shape: {shap_values.shape}"
    )

    return explainer, shap_values, X_test


# ============================================================
# SAVE SHAP VALUES
# ============================================================

def save_shap_values(
    shap_values,
    X_test,
    test_df,
):

    output = test_df[
        ["Subject", "Label"]
    ].copy()

    output["MRI_PD_Probability"] = X_test[
        "MRI_PD_Probability"
    ].values

    output["Clinical_PD_Probability"] = X_test[
        "Clinical_PD_Probability"
    ].values

    output["MRI_SHAP"] = shap_values[:, 0]

    output["Clinical_SHAP"] = shap_values[:, 1]

    output["Base_Value"] = (
        output["MRI_SHAP"] * 0
    )

    output["Absolute_MRI_SHAP"] = (
        output["MRI_SHAP"].abs()
    )

    output["Absolute_Clinical_SHAP"] = (
        output["Clinical_SHAP"].abs()
    )

    path = OUT_DIR / "fusion_shap_values.csv"

    output.to_csv(path, index=False)

    print(f"\nSaved: {path}")

    return output


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def save_feature_importance(shap_values):

    mean_abs = np.abs(shap_values).mean(axis=0)

    total = mean_abs.sum()

    contribution = (
        mean_abs / total * 100
    )

    importance = pd.DataFrame(
        {
            "Feature": FEATURES,
            "Mean_Absolute_SHAP": mean_abs,
            "Contribution_Percent": contribution,
        }
    )

    importance = importance.sort_values(
        "Mean_Absolute_SHAP",
        ascending=False,
    )

    path = (
        OUT_DIR
        / "fusion_shap_feature_importance.csv"
    )

    importance.to_csv(path, index=False)

    print("\n=== FUSION SHAP FEATURE IMPORTANCE ===")

    print(
        importance.to_string(index=False)
    )

    print(
        f"\nTotal contribution: "
        f"{importance['Contribution_Percent'].sum():.6f}%"
    )

    print(f"\nSaved: {path}")

    return importance


# ============================================================
# SHAP BAR PLOT
# ============================================================

def save_bar_plot(shap_values, X_test):

    plt.figure(figsize=(7, 5))

    shap.summary_plot(
        shap_values,
        X_test,
        plot_type="bar",
        show=False,
    )

    plt.title(
        "Fusion SHAP Feature Importance"
    )

    plt.tight_layout()

    path = (
        OUT_DIR
        / "fusion_shap_bar.png"
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {path}")


# ============================================================
# SHAP BEESWARM
# ============================================================

def save_beeswarm(shap_values, X_test):

    plt.figure(figsize=(8, 5))

    shap.summary_plot(
        shap_values,
        X_test,
        show=False,
    )

    plt.title(
        "Fusion SHAP Summary"
    )

    plt.tight_layout()

    path = (
        OUT_DIR
        / "fusion_shap_beeswarm.png"
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {path}")


# ============================================================
# MRI VS CLINICAL CONTRIBUTION
# ============================================================

def save_modality_comparison(shap_values):

    mean_abs = np.abs(shap_values).mean(axis=0)

    total = mean_abs.sum()

    percentages = (
        mean_abs / total * 100
    )

    plt.figure(figsize=(7, 5))

    plt.bar(
        [
            "MRI",
            "Clinical",
        ],
        percentages,
    )

    plt.ylabel(
        "Mean Absolute SHAP Contribution (%)"
    )

    plt.title(
        "Fusion Contribution: MRI vs Clinical"
    )

    plt.ylim(0, 100)

    for i, value in enumerate(percentages):

        plt.text(
            i,
            value + 1,
            f"{value:.2f}%",
            ha="center",
        )

    plt.tight_layout()

    path = (
        OUT_DIR
        / "mri_vs_clinical_shap_contribution.png"
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {path}")


# ============================================================
# SELECTED SUBJECT WATERFALLS
# ============================================================

def save_selected_waterfalls(
    explainer,
    shap_values,
    X_test,
    test_df,
):

    for subject in SELECTED_SUBJECTS:

        matches = test_df.index[
            test_df["Subject"] == subject
        ].tolist()

        if not matches:

            print(
                f"Subject {subject} not found "
                "in Fusion test set."
            )

            continue

        idx = matches[0]

        explanation = shap.Explanation(
            values=shap_values[idx],
            base_values=explainer.expected_value,
            data=X_test.iloc[idx].values,
            feature_names=FEATURES,
        )

        plt.figure(figsize=(9, 5))

        shap.plots.waterfall(
            explanation,
            show=False,
        )

        plt.title(
            f"Fusion SHAP Explanation - Subject {subject}"
        )

        plt.tight_layout()

        path = (
            OUT_DIR
            / f"fusion_shap_waterfall_{subject}.png"
        )

        plt.savefig(
            path,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close()

        print(f"Saved: {path}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("==========================================")
    print("NEUROINSIGHT FUSION SHAP ANALYSIS")
    print("==========================================")

    print("\nLoading Fusion data...")

    df = load_data()

    print(
        f"Aligned subjects: {len(df)}"
    )

    print(
        f"Train subjects: "
        f"{(df['Split'] == 'train').sum()}"
    )

    print(
        f"Test subjects: "
        f"{(df['Split'] == 'test').sum()}"
    )

    print(
        "Features:",
        FEATURES,
    )

    mlp, train_df, test_df = train_fusion_model(
        df
    )

    print(
        "\nFusion MLP trained successfully."
    )

    verify_model(
        mlp,
        df,
    )

    explainer, shap_values, X_test = calculate_shap(
        mlp,
        train_df,
        test_df,
    )

    save_shap_values(
        shap_values,
        X_test,
        test_df,
    )

    save_feature_importance(
        shap_values
    )

    save_bar_plot(
        shap_values,
        X_test,
    )

    save_beeswarm(
        shap_values,
        X_test,
    )

    save_modality_comparison(
        shap_values
    )

    save_selected_waterfalls(
        explainer,
        shap_values,
        X_test,
        test_df,
    )

    print("\n==========================================")
    print("FUSION SHAP ANALYSIS COMPLETE")
    print("==========================================")

    print(
        f"\nAll outputs saved in:\n{OUT_DIR}"
    )


if __name__ == "__main__":
    main()