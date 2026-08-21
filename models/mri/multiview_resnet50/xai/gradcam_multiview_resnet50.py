import os
import numpy as np
import tensorflow as tf
import cv2


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "MRI_Model",
    "multiview_cnn_results",
    "best_multiview_resnet50.keras"
)

NPY_DIR = os.path.join(
    BASE_DIR,
    "MRI_Model",
    "cnn_dataset",
    "normalized"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "MRI_Model",
    "xai",
    "gradcam"
)

IMG_SIZE = 160
SLICES_PER_VIEW = 5

# Grad-CAM visualization settings
HEATMAP_ALPHA = 0.45
HEATMAP_THRESHOLD = 0.20

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("LOADING TRAINED MULTI-VIEW RESNET50 MODEL")
print("=" * 70)

model = tf.keras.models.load_model(
    MODEL_PATH
)

# Explicitly call/build the Sequential model
dummy_input = tf.zeros(
    (1, IMG_SIZE, IMG_SIZE, 3),
    dtype=tf.float32
)

dummy_output = model(
    dummy_input,
    training=False
)

print(
    "Model loaded successfully."
)

print(
    "Input shape :",
    model.inputs[0].shape
)

print(
    "Output shape:",
    dummy_output.shape
)


# ============================================================
# GET NESTED RESNET50
# ============================================================

resnet50 = model.get_layer(
    "resnet50"
)

print(
    "ResNet50 output:",
    resnet50.output.shape
)


# ============================================================
# GRAD-CAM TARGET
# ============================================================

target_layer = resnet50.get_layer(
    "conv5_block3_out"
)

print(
    "Grad-CAM target:",
    target_layer.name
)

print(
    "Target output shape:",
    target_layer.output.shape
)


# ============================================================
# CLASSIFIER LAYERS
# ============================================================

classifier_layers = model.layers[1:]

print()
print("Classifier layers:")

for layer in classifier_layers:

    print(
        " -",
        layer.name,
        "|",
        type(layer).__name__
    )


# ============================================================
# NPY PATH
# ============================================================

def get_npy_path(
    mri_file
):

    filename = os.path.basename(
        mri_file
    )

    filename = filename.replace(
        ".nii.gz",
        ".npy"
    )

    return os.path.join(
        NPY_DIR,
        filename
    )


# ============================================================
# SAME SLICE SELECTION AS TRAINING
# ============================================================

def get_indices(
    length
):

    start = int(
        length * 0.25
    )

    end = int(
        length * 0.75
    )

    return np.linspace(
        start,
        end,
        SLICES_PER_VIEW
    ).astype(
        int
    )


# ============================================================
# SAME NORMALIZATION AS TRAINING
# ============================================================

def normalize_slice(
    img
):

    img = np.nan_to_num(
        img
    )

    low = np.percentile(
        img,
        1
    )

    high = np.percentile(
        img,
        99
    )

    if high <= low:

        img = np.zeros_like(
            img
        )

    else:

        img = np.clip(
            img,
            low,
            high
        )

        img = (
            img - low
        ) / (
            high - low
        )

    # 0-1 -> 0-255

    img = (
        img * 255.0
    ).astype(
        np.float32
    )

    # Add channel

    img = tf.convert_to_tensor(
        img[..., np.newaxis],
        dtype=tf.float32
    )

    # Resize

    img = tf.image.resize(
        img,
        (
            IMG_SIZE,
            IMG_SIZE
        )
    )

    img = img.numpy()

    # Grayscale -> RGB

    img = np.repeat(
        img,
        3,
        axis=-1
    )

    return img


# ============================================================
# EXTRACT 15 MULTI-VIEW SLICES
# ============================================================

def extract_multiview_slices(
    npy_path
):

    volume = np.load(
        npy_path
    ).astype(
        np.float32
    )

    if volume.ndim != 3:

        raise ValueError(
            f"Expected 3D MRI, "
            f"got {volume.shape}"
        )

    x, y, z = volume.shape

    slices = []


    # ========================================================
    # AXIAL
    # ========================================================

    axial_indices = get_indices(
        z
    )

    for idx in axial_indices:

        raw = volume[
            :,
            :,
            idx
        ]

        processed = normalize_slice(
            raw
        )

        slices.append({

            "view": "axial",

            "index": int(idx),

            "raw": raw,

            "processed": processed

        })


    # ========================================================
    # CORONAL
    # ========================================================

    coronal_indices = get_indices(
        y
    )

    for idx in coronal_indices:

        raw = volume[
            :,
            idx,
            :
        ]

        processed = normalize_slice(
            raw
        )

        slices.append({

            "view": "coronal",

            "index": int(idx),

            "raw": raw,

            "processed": processed

        })


    # ========================================================
    # SAGITTAL
    # ========================================================

    sagittal_indices = get_indices(
        x
    )

    for idx in sagittal_indices:

        raw = volume[
            idx,
            :,
            :
        ]

        processed = normalize_slice(
            raw
        )

        slices.append({

            "view": "sagittal",

            "index": int(idx),

            "raw": raw,

            "processed": processed

        })


    if len(slices) != 15:

        raise RuntimeError(
            f"Expected 15 slices, "
            f"got {len(slices)}"
        )

    return slices


# ============================================================
# CLASSIFIER FROM RESNET FEATURES
# ============================================================

def classifier_from_features(
    features
):

    x = features

    for layer in classifier_layers:

        x = layer(
            x,
            training=False
        )

    return x


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

def generate_gradcam(
    processed_slice
):

    input_tensor = tf.convert_to_tensor(
        processed_slice[
            np.newaxis,
            ...
        ],
        dtype=tf.float32
    )

    # --------------------------------------------------------
    # Model ending at target convolutional layer and
    # ResNet50 feature output
    # --------------------------------------------------------

    resnet_grad_model = tf.keras.models.Model(
        inputs=resnet50.inputs,
        outputs=[
            target_layer.output,
            resnet50.output
        ]
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with tf.GradientTape() as tape:

        conv_output, features = (
            resnet_grad_model(
                input_tensor,
                training=False
            )
        )

        prediction = classifier_from_features(
            features
        )

        # Binary sigmoid output
        score = prediction[:, 0]

    # --------------------------------------------------------
    # Gradient
    # --------------------------------------------------------

    gradients = tape.gradient(
        score,
        conv_output
    )

    if gradients is None:

        raise RuntimeError(
            "Gradients are None. "
            "Could not connect the PD output "
            "to conv5_block3_out."
        )

    # --------------------------------------------------------
    # Global average pooling
    # --------------------------------------------------------

    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    conv_features = conv_output[0]

    weights = weights[0]

    # --------------------------------------------------------
    # Weighted activation maps
    # --------------------------------------------------------

    cam = tf.reduce_sum(
        conv_features * weights,
        axis=-1
    )

    # Positive contribution only

    cam = tf.nn.relu(
        cam
    )

    # Normalize

    cam_max = tf.reduce_max(
        cam
    )

    cam = cam / (
        cam_max + 1e-8
    )

    cam = cam.numpy()

    # --------------------------------------------------------
    # 5 x 5 -> 160 x 160
    # --------------------------------------------------------

    cam = cv2.resize(
        cam,
        (
            IMG_SIZE,
            IMG_SIZE
        ),
        interpolation=cv2.INTER_LINEAR
    )

    probability = float(
        prediction[0, 0].numpy()
    )

    return (
        cam,
        probability
    )


# ============================================================
# CREATE BRAIN MASK
# ============================================================

def create_brain_mask(
    original_slice
):

    img = np.asarray(
        original_slice,
        dtype=np.float32
    )

    if img.ndim == 3:

        img = img[..., 0]

    # Normalize

    img = (
        img - img.min()
    )

    max_value = img.max()

    if max_value > 0:

        img = (
            img / max_value
        )

    img_uint8 = (
        img * 255
    ).astype(
        np.uint8
    )

    # Resize

    img_uint8 = cv2.resize(
        img_uint8,
        (
            IMG_SIZE,
            IMG_SIZE
        ),
        interpolation=cv2.INTER_LINEAR
    )

    # Otsu threshold

    _, mask = cv2.threshold(
        img_uint8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Morphological cleanup

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Keep largest component

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    if num_labels > 1:

        largest_label = (
            1
            +
            np.argmax(
                stats[
                    1:,
                    cv2.CC_STAT_AREA
                ]
            )
        )

        mask = np.where(
            labels == largest_label,
            255,
            0
        ).astype(
            np.uint8
        )

    return mask


# ============================================================
# IMPROVED HEATMAP OVERLAY
# ============================================================

def overlay_heatmap(
    original_slice,
    heatmap,
    alpha=HEATMAP_ALPHA,
    threshold=HEATMAP_THRESHOLD
):

    # ========================================================
    # ORIGINAL MRI
    # ========================================================

    img = np.asarray(
        original_slice,
        dtype=np.float32
    )

    if img.ndim == 3:

        img = img[..., 0]

    img = (
        img - img.min()
    )

    max_value = img.max()

    if max_value > 0:

        img = (
            img / max_value
        )

    img = (
        img * 255
    ).clip(
        0,
        255
    ).astype(
        np.uint8
    )

    # IMPORTANT:
    # Make original exactly 160 x 160.

    img = cv2.resize(
        img,
        (
            IMG_SIZE,
            IMG_SIZE
        ),
        interpolation=cv2.INTER_LINEAR
    )

    img_rgb = cv2.cvtColor(
        img,
        cv2.COLOR_GRAY2RGB
    )


    # ========================================================
    # BRAIN MASK
    # ========================================================

    brain_mask = create_brain_mask(
        original_slice
    )

    brain_mask_float = (
        brain_mask.astype(
            np.float32
        )
        /
        255.0
    )


    # ========================================================
    # HEATMAP
    # ========================================================

    heatmap = np.asarray(
        heatmap,
        dtype=np.float32
    )

    heatmap = cv2.resize(
        heatmap,
        (
            IMG_SIZE,
            IMG_SIZE
        ),
        interpolation=cv2.INTER_LINEAR
    )

    heatmap = (
        heatmap - heatmap.min()
    )

    heatmap_max = heatmap.max()

    if heatmap_max > 0:

        heatmap = (
            heatmap / heatmap_max
        )


    # ========================================================
    # THRESHOLD
    # ========================================================

    heatmap = np.where(
        heatmap >= threshold,
        heatmap,
        0
    )


    # ========================================================
    # BACKGROUND SUPPRESSION
    # ========================================================

    heatmap = (
        heatmap
        *
        brain_mask_float
    )


    # ========================================================
    # SMOOTH
    # ========================================================

    heatmap = cv2.GaussianBlur(
        heatmap,
        (0, 0),
        sigmaX=7
    )

    heatmap = (
        heatmap - heatmap.min()
    )

    heatmap_max = heatmap.max()

    if heatmap_max > 0:

        heatmap = (
            heatmap / heatmap_max
        )


    # ========================================================
    # COLOR MAP
    # ========================================================

    heatmap_uint8 = (
        heatmap * 255
    ).clip(
        0,
        255
    ).astype(
        np.uint8
    )

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )


    # ========================================================
    # ACTIVATION MASK
    # ========================================================

    activation_mask = (
        heatmap > 0
    ).astype(
        np.float32
    )

    activation_mask = cv2.GaussianBlur(
        activation_mask,
        (0, 0),
        sigmaX=3
    )


    # ========================================================
    # BLEND
    # ========================================================

    img_float = (
        img_rgb.astype(
            np.float32
        )
    )

    heat_float = (
        heatmap_color.astype(
            np.float32
        )
    )

    mask_3d = (
        activation_mask[
            ...,
            np.newaxis
        ]
    )

    blended = (
        img_float
        *
        (
            1
            -
            alpha * mask_3d
        )
        +
        heat_float
        *
        (
            alpha * mask_3d
        )
    )

    blended = blended.clip(
        0,
        255
    ).astype(
        np.uint8
    )

    return blended


# ============================================================
# ADD CLEAN PANEL HEADER
# ============================================================

def add_text(
    image,
    view,
    slice_index,
    probability
):

    result = image.copy()

    # Header area

    result = cv2.copyMakeBorder(
        result,
        75,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(20, 20, 20)
    )

    # View name

    cv2.putText(
        result,
        view.upper(),
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Slice + probability

    text = (
        f"Slice: {slice_index}   "
        f"PD: {probability:.3f}"
    )

    cv2.putText(
        result,
        text,
        (12, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA
    )

    return result


# ============================================================
# CREATE LARGE 3-VIEW GRID
# ============================================================

def create_gradcam_grid(
    selected,
    subject
):

    panels = []

    for view in [
        "axial",
        "coronal",
        "sagittal"
    ]:

        result = selected[view]

        overlay = overlay_heatmap(
            result["raw"],
            result["heatmap"],
            alpha=HEATMAP_ALPHA,
            threshold=HEATMAP_THRESHOLD
        )

        # ----------------------------------------------------
        # Enlarge panel
        # ----------------------------------------------------

        overlay = cv2.resize(
            overlay,
            (
                300,
                300
            ),
            interpolation=cv2.INTER_CUBIC
        )

        overlay = add_text(
            overlay,
            view,
            result["index"],
            result["probability"]
        )

        panels.append(
            overlay
        )

    # ========================================================
    # SIDE-BY-SIDE
    # ========================================================

    grid = np.hstack(
        panels
    )

    # ========================================================
    # SUBJECT TITLE
    # ========================================================

    grid = cv2.copyMakeBorder(
        grid,
        65,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(10, 10, 10)
    )

    cv2.putText(
        grid,
        f"Subject {subject} - MRI Grad-CAM",
        (20, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return grid


# ============================================================
# SAVE IMAGE
# ============================================================

def save_image(
    image,
    filename
):

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    success = cv2.imwrite(
        output_path,
        cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )
    )

    if not success:

        raise RuntimeError(
            f"Could not save:\n"
            f"{output_path}"
        )

    print(
        "Saved:",
        output_path
    )


# ============================================================
# PROCESS ONE MRI
# ============================================================

def process_mri(
    subject,
    mri_file
):

    print()
    print("=" * 70)

    print(
        "SUBJECT:",
        subject
    )

    print(
        "MRI FILE:",
        mri_file
    )

    print("=" * 70)


    # ========================================================
    # NPY PATH
    # ========================================================

    npy_path = get_npy_path(
        mri_file
    )

    print(
        "NPY PATH:",
        npy_path
    )

    if not os.path.exists(
        npy_path
    ):

        raise FileNotFoundError(
            f"MRI NPY file not found:\n"
            f"{npy_path}"
        )


    # ========================================================
    # EXTRACT SLICES
    # ========================================================

    slices = extract_multiview_slices(
        npy_path
    )

    print(
        f"Total slices: {len(slices)}"
    )


    # ========================================================
    # PROCESS ALL 15 SLICES
    # ========================================================

    results = []

    for i, item in enumerate(
        slices,
        start=1
    ):

        print(
            f"\nProcessing "
            f"{i}/15: "
            f"{item['view']} "
            f"slice {item['index']}"
        )

        heatmap, probability = (
            generate_gradcam(
                item["processed"]
            )
        )

        print(
            f"PD probability: "
            f"{probability:.6f}"
        )

        results.append({

            "view":
                item["view"],

            "index":
                item["index"],

            "raw":
                item["raw"],

            "heatmap":
                heatmap,

            "probability":
                probability

        })


    # ========================================================
    # SELECT HIGHEST-PROBABILITY SLICE PER VIEW
    # ========================================================

    selected = {}

    for view in [
        "axial",
        "coronal",
        "sagittal"
    ]:

        view_results = [

            r
            for r in results

            if r["view"] == view

        ]

        best = max(
            view_results,
            key=lambda r: r["probability"]
        )

        selected[view] = best

        print()
        print(
            f"SELECTED {view.upper()}"
        )

        print(
            "Slice:",
            best["index"]
        )

        print(
            "PD probability:",
            f"{best['probability']:.6f}"
        )


    # ========================================================
    # SAVE INDIVIDUAL IMAGES
    # ========================================================

    for view in [
        "axial",
        "coronal",
        "sagittal"
    ]:

        result = selected[view]

        overlay = overlay_heatmap(
            result["raw"],
            result["heatmap"],
            alpha=HEATMAP_ALPHA,
            threshold=HEATMAP_THRESHOLD
        )

        filename = (
            f"subject_{subject}_"
            f"{view}_"
            f"slice_{result['index']}.png"
        )

        save_image(
            overlay,
            filename
        )


    # ========================================================
    # CREATE GRID
    # ========================================================

    grid = create_gradcam_grid(
        selected,
        subject
    )

    grid_filename = (
        f"subject_{subject}_"
        "gradcam_grid.png"
    )

    grid_path = os.path.join(
        OUTPUT_DIR,
        grid_filename
    )

    success = cv2.imwrite(
        grid_path,
        grid
    )

    if not success:

        raise RuntimeError(
            f"Could not save:\n"
            f"{grid_path}"
        )

    print(
        "Saved:",
        grid_path
    )


    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 70)
    print("GRAD-CAM COMPLETE")
    print("=" * 70)

    print(
        "Output directory:"
    )

    print(
        OUTPUT_DIR
    )


# ============================================================
# BATCH PROCESS ALL TEST SUBJECTS
# ============================================================

if __name__ == "__main__":

    import pandas as pd

    TEST_CSV = os.path.join(
        BASE_DIR,
        "MRI_Model",
        "multimodal_split_v2",
        "test.csv"
    )

    print()
    print("=" * 70)
    print("LOADING TEST DATA")
    print("=" * 70)

    test_df = pd.read_csv(
        TEST_CSV
    )

    # Make Subject consistent
    test_df["Subject"] = (
        test_df["Subject"]
        .astype(str)
        .str.strip()
    )

    # Remove 3102 because we already generated it
    test_df = test_df[
        test_df["Subject"] != "3102"
    ]

    # One MRI record per subject
    subject_df = (
        test_df[
            [
                "Subject",
                "MRI_File",
                "Label"
            ]
        ]
        .drop_duplicates(
            subset=["Subject"]
        )
        .reset_index(
            drop=True
        )
    )

    print(
        "Remaining test subjects:",
        len(subject_df)
    )

    print()

    # ========================================================
    # PROCESS EACH SUBJECT
    # ========================================================

    for position, row in subject_df.iterrows():

        subject = str(
            row["Subject"]
        )

        mri_file = str(
            row["MRI_File"]
        )

        label = int(
            row["Label"]
        )

        print()
        print("#" * 70)

        print(
            f"TEST SUBJECT "
            f"{position + 1}/{len(subject_df)}"
        )

        print(
            "Subject:",
            subject
        )

        print(
            "MRI:",
            mri_file
        )

        print(
            "True label:",
            label
        )

        print(
            "#" * 70
        )

        try:

            process_mri(
                subject,
                mri_file
            )

        except Exception as e:

            print()
            print(
                "ERROR PROCESSING SUBJECT:",
                subject
            )

            print(
                "MRI:",
                mri_file
            )

            print(
                "Error:",
                repr(e)
            )

            print(
                "Skipping this subject "
                "and continuing..."
            )

            continue


    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 70)
    print("ALL REMAINING TEST SUBJECTS PROCESSED")
    print("=" * 70)

    print(
        "Output directory:"
    )

    print(
        OUTPUT_DIR
    )