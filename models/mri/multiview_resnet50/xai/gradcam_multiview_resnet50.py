import os
import cv2
import numpy as np
import tensorflow as tf


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "mri",
    "multiview_resnet50",
    "best_multiview_resnet50.keras"
)

NPY_DIR = os.path.join(
    BASE_DIR,
    "models",
    "mri",
    "test_data",
    "normalized"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "models",
    "mri",
    "multiview_resnet50",
    "xai",
    "gradcam"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SELECTED SUBJECTS
# ============================================================

SELECTED_SUBJECTS = {
    "174151": {
        "label": "Correct PD",
        "mri_file": "I10650738.npy"
    },
    "137842": {
        "label": "Correct PD",
        "mri_file": "I1580943.npy"
    },
    "3517": {
        "label": "Correct Control",
        "mri_file": "I260386.npy"
    },
    "4010": {
        "label": "Correct Control",
        "mri_file": "I288213.npy"
    },
    "3178": {
        "label": "MRI Missed PD",
        "mri_file": "I315639.npy"
    },
    "293727": {
        "label": "MRI False Alarm",
        "mri_file": "I10368714.npy"
    }
}


# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = 160
TARGET_LAYER_NAME = "conv5_block3_out"


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING TRAINED MULTI-VIEW RESNET50 MODEL")
print("=" * 70)

print("\nMODEL PATH:")
print(MODEL_PATH)

print("\nNPY DIRECTORY:")
print(NPY_DIR)

print("\nOUTPUT DIRECTORY:")
print(OUTPUT_DIR)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}"
    )

if not os.path.exists(NPY_DIR):
    raise FileNotFoundError(
        f"\nNPY directory not found:\n{NPY_DIR}"
    )

model = tf.keras.models.load_model(MODEL_PATH)

print("\nModel loaded successfully.")
print("Input shape :", model.input_shape)
print("Output shape:", model.output_shape)


# ============================================================
# IDENTIFY NESTED RESNET50
# ============================================================

resnet = model.layers[0]

print("\nResNet50 output:", resnet.output_shape)

target_layer = resnet.get_layer(TARGET_LAYER_NAME)

print("Grad-CAM target:", TARGET_LAYER_NAME)

try:
    target_shape = target_layer.output.shape
except Exception:
    target_shape = "Unknown"

print("Target output shape:", target_shape)


# ============================================================
# CLASSIFIER LAYERS
# ============================================================

classifier_layers = model.layers[1:]

print("\nClassifier layers:")

for layer in classifier_layers:
    print(f" - {layer.name} | {layer.__class__.__name__}")


# ============================================================
# GRAD-CAM MODEL
# ============================================================
#
# IMPORTANT:
#
# We do NOT use:
#
#     model.input
#
# because the loaded Sequential model can raise:
#
#     AttributeError:
#     The layer sequential has never been called
#
# Instead, we create a separate gradient model from the
# nested ResNet50 itself.
#
# ============================================================

grad_resnet_model = tf.keras.Model(
    inputs=resnet.input,
    outputs=[
        target_layer.output,
        resnet.output
    ]
)

print("\nGrad-CAM gradient model created successfully.")


# ============================================================
# APPLY CLASSIFIER
# ============================================================

def apply_classifier(resnet_output):
    """
    Pass the ResNet50 feature output through the
    original classifier layers.

    This reproduces the trained model's prediction
    without modifying the trained model.
    """

    x = resnet_output

    for layer in classifier_layers:
        x = layer(x, training=False)

    return x


# ============================================================
# GET NPY PATH
# ============================================================

def get_npy_path(mri_file):

    path = os.path.join(
        NPY_DIR,
        mri_file
    )

    return path


# ============================================================
# LOAD MRI
# ============================================================

def load_mri(npy_path):

    data = np.load(npy_path)

    print("\nLoaded MRI:")
    print("  Shape:", data.shape)
    print("  Min  :", np.min(data))
    print("  Max  :", np.max(data))

    return data


# ============================================================
# EXTRACT MULTI-VIEW SLICES
# ============================================================

def extract_multiview_slices(volume):

    """
    Extract representative slices from the 3D MRI volume.

    Produces:
        axial
        coronal
        sagittal

    Each view contains 5 slices.
    """

    if volume.ndim != 3:
        raise ValueError(
            f"Expected 3D MRI volume, got shape {volume.shape}"
        )

    depth, height, width = volume.shape

    center_d = depth // 2
    center_h = height // 2
    center_w = width // 2

    offsets = [-20, -10, 0, 10, 20]

    slices = []

    for offset in offsets:

        # -----------------------------
        # AXIAL
        # -----------------------------

        d = np.clip(
            center_d + offset,
            0,
            depth - 1
        )

        axial = volume[d, :, :]

        # -----------------------------
        # CORONAL
        # -----------------------------

        h = np.clip(
            center_h + offset,
            0,
            height - 1
        )

        coronal = volume[:, h, :]

        # -----------------------------
        # SAGITTAL
        # -----------------------------

        w = np.clip(
            center_w + offset,
            0,
            width - 1
        )

        sagittal = volume[:, :, w]

        slices.append({
            "axial": axial,
            "coronal": coronal,
            "sagittal": sagittal,
            "slice_offset": offset
        })

    return slices


# ============================================================
# PREPROCESS SLICE
# ============================================================

def preprocess_slice(slice_2d):

    image = np.asarray(
        slice_2d,
        dtype=np.float32
    )

    # Handle NaN / infinity
    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # Normalize to 0-1
    min_val = np.min(image)
    max_val = np.max(image)

    if max_val > min_val:

        image = (
            image - min_val
        ) / (
            max_val - min_val
        )

    else:

        image = np.zeros_like(image)

    # Resize
    image = cv2.resize(
        image,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_LINEAR
    )

    # Convert grayscale -> RGB
    image = np.stack(
        [image, image, image],
        axis=-1
    )

    return image.astype(np.float32)


# ============================================================
# BRAIN MASK
# ============================================================

def create_brain_mask(original_slice):

    image = np.asarray(
        original_slice,
        dtype=np.float32
    )

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    min_val = np.min(image)
    max_val = np.max(image)

    if max_val > min_val:

        image = (
            (image - min_val)
            / (max_val - min_val)
            * 255
        )

    else:

        image = np.zeros_like(image)

    image = image.astype(np.uint8)

    image = cv2.resize(
        image,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_LINEAR
    )

    # Otsu threshold
    _, mask = cv2.threshold(
        image,
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

    # Keep largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    if num_labels > 1:

        largest_label = 1 + np.argmax(
            stats[1:, cv2.CC_STAT_AREA]
        )

        mask = np.where(
            labels == largest_label,
            255,
            0
        ).astype(np.uint8)

    return mask


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

def generate_gradcam(processed_image):

    """
    Generate Grad-CAM for one MRI slice.

    Returns:
        probability
        heatmap
        raw CAM
    """

    image_tensor = tf.convert_to_tensor(
        processed_image[np.newaxis, ...],
        dtype=tf.float32
    )

    with tf.GradientTape() as tape:

        target_activation, resnet_output = grad_resnet_model(
            image_tensor,
            training=False
        )

        tape.watch(target_activation)

        prediction = apply_classifier(
            resnet_output
        )

        score = prediction[:, 0]

    gradients = tape.gradient(
        score,
        target_activation
    )

    if gradients is None:

        raise RuntimeError(
            "\nGradients are None.\n"
            "Could not connect the PD output "
            f"to {TARGET_LAYER_NAME}."
        )

    # Average gradients over spatial dimensions
    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    # First image in batch
    feature_maps = target_activation[0]

    weights = weights[0]

    # Weighted combination of feature maps
    cam = tf.reduce_sum(
        feature_maps * weights,
        axis=-1
    )

    # ReLU
    cam = tf.nn.relu(cam)

    # Normalize
    cam_max = tf.reduce_max(cam)

    cam = cam / (
        cam_max + 1e-8
    )

    cam = cam.numpy()

    # Resize 5x5 -> 160x160
    cam = cv2.resize(
        cam,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_LINEAR
    )

    probability = float(
        prediction.numpy()[0][0]
    )

    return probability, cam


# ============================================================
# CREATE OVERLAY
# ============================================================

def overlay_heatmap(
    original_slice,
    heatmap
):

    # Prepare original MRI
    original = np.asarray(
        original_slice,
        dtype=np.float32
    )

    original = np.nan_to_num(
        original,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    min_val = np.min(original)
    max_val = np.max(original)

    if max_val > min_val:

        original = (
            (original - min_val)
            / (max_val - min_val)
            * 255
        )

    else:

        original = np.zeros_like(original)

    original = cv2.resize(
        original.astype(np.uint8),
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_LINEAR
    )

    original_rgb = cv2.cvtColor(
        original,
        cv2.COLOR_GRAY2RGB
    )

    # Brain mask
    brain_mask = create_brain_mask(
        original_slice
    )

    # Normalize heatmap
    heatmap = np.nan_to_num(
        heatmap,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    heatmap = np.clip(
        heatmap,
        0,
        1
    )

    # Suppress weak activation
    heatmap_threshold = 0.20

    heatmap_mask = (
        heatmap >= heatmap_threshold
    ).astype(np.uint8) * 255

    # Combine with brain mask
    combined_mask = cv2.bitwise_and(
        heatmap_mask,
        brain_mask
    )

    # Smooth heatmap
    heatmap = cv2.GaussianBlur(
        heatmap,
        (0, 0),
        sigmaX=3
    )

    heatmap_uint8 = (
        heatmap * 255
    ).astype(np.uint8)

    # Apply color map
    colored_heatmap = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    colored_heatmap = cv2.cvtColor(
        colored_heatmap,
        cv2.COLOR_BGR2RGB
    )

    # Keep only brain + activated area
    activation = np.zeros_like(
        colored_heatmap
    )

    activation[
        combined_mask > 0
    ] = colored_heatmap[
        combined_mask > 0
    ]

    # Blend
    overlay = cv2.addWeighted(
        original_rgb,
        0.65,
        activation,
        0.35,
        0
    )

    return overlay


# ============================================================
# ADD TEXT
# ============================================================

def add_text(
    image,
    view,
    slice_index,
    probability
):

    output = image.copy()

    text1 = f"{view.upper()} | Slice {slice_index}"

    text2 = f"PD Probability: {probability:.3f}"

    cv2.putText(
        output,
        text1,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        output,
        text2,
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return output


# ============================================================
# CREATE SUBJECT GRID
# ============================================================

def create_gradcam_grid(
    selected_images,
    subject
):

    panels = []

    for view in [
        "axial",
        "coronal",
        "sagittal"
    ]:

        if view not in selected_images:
            continue

        image = selected_images[view]

        image = cv2.resize(
            image,
            (300, 300),
            interpolation=cv2.INTER_AREA
        )

        panels.append(image)

    if len(panels) == 0:
        return None

    grid = np.hstack(
        panels
    )

    title_height = 60

    canvas = np.zeros(
        (
            300 + title_height,
            grid.shape[1],
            3
        ),
        dtype=np.uint8
    )

    canvas[
        title_height:,
        :
    ] = grid

    title = (
        f"Subject {subject} - MRI Grad-CAM"
    )

    cv2.putText(
        canvas,
        title,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return canvas


# ============================================================
# PROCESS ONE SUBJECT
# ============================================================

def process_subject(
    subject,
    information
):

    print("\n")
    print("=" * 70)
    print(f"PROCESSING SUBJECT: {subject}")
    print("=" * 70)

    print(
        f"Clinical/MRI interpretation: "
        f"{information['label']}"
    )

    mri_file = information["mri_file"]

    npy_path = get_npy_path(
        mri_file
    )

    print("\nMRI file:")
    print(npy_path)

    if not os.path.exists(npy_path):

        print(
            "\nERROR: MRI NPY file not found."
        )

        return

    volume = load_mri(
        npy_path
    )

    slices = extract_multiview_slices(
        volume
    )

    best_images = {}

    best_probabilities = {}

    best_indices = {}

    for slice_index, item in enumerate(slices):

        offset = item["slice_offset"]

        print(
            f"\nProcessing slice "
            f"{slice_index + 1}/5 "
            f"(offset {offset})"
        )

        for view in [
            "axial",
            "coronal",
            "sagittal"
        ]:

            original_slice = item[view]

            processed = preprocess_slice(
                original_slice
            )

            probability, cam = generate_gradcam(
                processed
            )

            overlay = overlay_heatmap(
                original_slice,
                cam
            )

            overlay = add_text(
                overlay,
                view,
                slice_index + 1,
                probability
            )

            print(
                f"  {view:10s} "
                f"PD probability = "
                f"{probability:.4f}"
            )

            # Save every individual slice
            filename = (
                f"subject_{subject}_"
                f"{view}_"
                f"slice_{slice_index + 1}.png"
            )

            output_path = os.path.join(
                OUTPUT_DIR,
                filename
            )

            cv2.imwrite(
                output_path,
                cv2.cvtColor(
                    overlay,
                    cv2.COLOR_RGB2BGR
                )
            )

            # Select highest probability slice
            if (
                view not in best_probabilities
                or probability >
                best_probabilities[view]
            ):

                best_probabilities[view] = (
                    probability
                )

                best_images[view] = overlay

                best_indices[view] = (
                    slice_index + 1
                )

    # ========================================================
    # SAVE BEST-VIEW IMAGES
    # ========================================================

    for view in [
        "axial",
        "coronal",
        "sagittal"
    ]:

        if view not in best_images:
            continue

        filename = (
            f"subject_{subject}_"
            f"{view}_BEST.png"
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            filename
        )

        cv2.imwrite(
            output_path,
            cv2.cvtColor(
                best_images[view],
                cv2.COLOR_RGB2BGR
            )
        )

    # ========================================================
    # SAVE GRID
    # ========================================================

    grid = create_gradcam_grid(
        best_images,
        subject
    )

    if grid is not None:

        grid_filename = (
            f"subject_{subject}_"
            f"gradcam_grid.png"
        )

        grid_path = os.path.join(
            OUTPUT_DIR,
            grid_filename
        )

        cv2.imwrite(
            grid_path,
            cv2.cvtColor(
                grid,
                cv2.COLOR_RGB2BGR
            )
        )

        print(
            "\nGrad-CAM grid saved:"
        )

        print(grid_path)

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\nBest Grad-CAM slices:")

    for view in [
        "axial",
        "coronal",
        "sagittal"
    ]:

        if view in best_probabilities:

            print(
                f"  {view:10s} "
                f"slice={best_indices[view]} "
                f"PD probability="
                f"{best_probabilities[view]:.4f}"
            )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("NEUROINSIGHT MRI GRAD-CAM")
    print("=" * 70)

    print(
        f"\nNumber of selected subjects: "
        f"{len(SELECTED_SUBJECTS)}"
    )

    print(
        f"Grad-CAM target layer: "
        f"{TARGET_LAYER_NAME}"
    )

    for subject, information in SELECTED_SUBJECTS.items():

        try:

            process_subject(
                subject,
                information
            )

        except Exception as e:

            print("\n")
            print("ERROR PROCESSING SUBJECT")
            print("-" * 50)
            print(f"Subject: {subject}")
            print(f"Error: {type(e).__name__}")
            print(f"Message: {e}")

            # Continue with remaining subjects
            continue

    print("\n")
    print("=" * 70)
    print("GRAD-CAM PROCESSING COMPLETE")
    print("=" * 70)

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    print("\nProcessed subjects:")

    for subject in SELECTED_SUBJECTS:
        print(
            f" - {subject} | "
            f"{SELECTED_SUBJECTS[subject]['label']}"
        )

    print("\n")