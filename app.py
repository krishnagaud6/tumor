# ============================================================
# BrainCare AI — PyTorch brain MRI classification demo
# Educational/research demo only; not a medical diagnostic tool.
# ============================================================

from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image
import torch.nn.functional as F

# -----------------------------
# 1. Page setup and visual style
# -----------------------------
st.set_page_config(
    page_title="BrainCare AI | MRI Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: linear-gradient(145deg, #f5f8ff 0%, #eef7f6 100%); color: #17253c; }
    .block-container { max-width: 1200px; padding-top: 2rem; padding-bottom: 3rem; }
    .brand { font-family: 'Manrope', sans-serif; font-size: 25px; font-weight: 800; color: #172d50; letter-spacing: -1px; }
    .brand span { color: #21a89b; }
    .eyebrow { color: #188c83; font-size: 12px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; }
    .hero { padding: 32px; border-radius: 24px; color: white; background: linear-gradient(120deg, #172d50, #245c76); margin: 14px 0 25px 0; }
    .hero h1 { color: white; font-family: 'Manrope', sans-serif; font-size: clamp(28px, 4vw, 43px); line-height: 1.15; margin: 12px 0; }
    .hero p { color: #dbe9f4; font-size: 15px; line-height: 1.8; }
    .panel { background: white; padding: 24px; border: 1px solid #e3eaf1; border-radius: 20px; margin-bottom: 18px; box-shadow: 0 7px 25px rgba(24, 48, 78, 0.04); }
    .metric { background: white; border: 1px solid #e3eaf1; border-radius: 17px; padding: 20px; min-height: 105px; }
    .metric-label { color: #708096; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-family: 'Manrope', sans-serif; font-size: 23px; font-weight: 800; color: #172d50; margin-top: 8px; }
    .mascot { text-align: center; font-size: 76px; padding: 14px; background: #f3f6ff; border-radius: 20px; margin-bottom: 12px; }
    div[data-testid="stFileUploader"] { background: #f6fafc; border: 2px dashed #a9c9d5; border-radius: 16px; padding: 16px; }
    .stButton > button { border-radius: 12px; min-height: 48px; font-weight: 700; }
    .footer { color: #77869a; text-align: center; font-size: 12px; padding: 25px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 2. MODEL SETTINGS — best-effort configuration from checkpoint
# The uploaded checkpoint contains weights only. We can infer two
# output units and reconstruct a matching parameter layout, but the
# true training labels and image preprocessing are not embedded.
# This setup shows Class 0 / Class 1, not a verified tumor diagnosis.
# ============================================================

MODEL_PATH = Path("tumor_cnn.pth")
MODEL_FORMAT = "state_dict"

# Best-effort assumption: a two-unit final Linear layer returns logits.
# The winning index is meaningful as a model class, but we do NOT know
# which index means tumor until the original label mapping is available.
OUTPUT_MODE = "two_class_logits"
CLASS_NAMES = ["Class 0 (unmapped)", "Class 1 (unmapped)"]

# Mascot behavior only activates if class labels are later mapped explicitly.
NO_TUMOR_LABELS = {"no tumor", "normal", "healthy", "no_tumour", "no tumor (verified)"}
TUMOR_LABELS = {"tumor", "tumour", "brain tumor", "positive", "tumor (verified)"}

# Best-effort assumption: two 2x2 pooling operations reduce 32x32 -> 8x8,
# matching the classifier's 32*8*8 = 2048 input features. The inferred
# model also includes adaptive pooling to 8x8, which is a no-op at this size.
# Training may have used different preprocessing; verify before relying on outputs.
IMAGE_SIZE = (224, 224)  # (width, height); demo default, not confirmed from training
IMAGE_MODE = "RGB"

# Best-effort assumption: scale pixel values to [0, 1]; no mean/std normalization.
NORMALIZE_MEAN = None
NORMALIZE_STD = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model_for_state_dict():
    """Build the exact architecture used during training.

    Your friend must provide model_definition.py with a create_model()
    function, or you must paste/import the original model class here.
    A state_dict contains weights, not the model architecture itself.
    """
    from model_definition import create_model
    return create_model()



@st.cache_resource

def load_model():
    """Load a model once and cache it across Streamlit reruns."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. Upload the model file and update MODEL_PATH."
        )

    if MODEL_FORMAT == "torchscript":
        # Use this when your friend exports using torch.jit.script/trace.
        model = torch.jit.load(str(MODEL_PATH), map_location=DEVICE)

    elif MODEL_FORMAT == "state_dict":
        # A state_dict contains tensors only. The matching architecture is required.
        model = build_model_for_state_dict()
        checkpoint = torch.load(
            str(MODEL_PATH),
            map_location="cpu",
            weights_only=True,  # safer than unpickling arbitrary Python objects
        )

        # Training code may save weights directly or inside a checkpoint dictionary.
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
        else:
            raise TypeError(
                "Expected a state_dict/checkpoint dictionary. Ask your friend how the model was saved."
            )

        # Some training scripts save keys with a 'module.' prefix from DataParallel.
        if state_dict and all(key.startswith("module.") for key in state_dict):
            state_dict = {key[len("module."):]: value for key, value in state_dict.items()}

        model.load_state_dict(state_dict)
        model = model.to(DEVICE)

    else:
        raise ValueError(
            "Set MODEL_FORMAT to 'torchscript' or 'state_dict' after confirming how the model was saved."
        )

    # eval() switches off training-only behavior such as dropout and training-mode batch norm.
    model = model.to(DEVICE)
    model.eval()
    return model


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image into the model input tensor [batch, channels, height, width]."""
    if IMAGE_MODE.upper() == "L":
        image = image.convert("L")
    elif IMAGE_MODE.upper() == "RGB":
        image = image.convert("RGB")
    else:
        raise ValueError("IMAGE_MODE must be 'RGB' or 'L'.")

    # Resize using the same dimensions as the training pipeline.
    image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)

    # PIL arrays are H x W x C (or H x W for grayscale).
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        array = array[:, :, np.newaxis]

    # Common training convention: scale image values from 0..255 to 0..1.
    # Remove this scaling if the model was trained with a different pipeline.
    array = array / 255.0

    # Change HWC to CHW, which is the usual PyTorch image layout.
    tensor = torch.from_numpy(array.transpose(2, 0, 1).copy()).float()
    tensor = tensor.unsqueeze(0)  # add batch dimension -> [1, C, H, W]

    # Optional normalization. Values must exactly match training preprocessing.
    if NORMALIZE_MEAN is not None or NORMALIZE_STD is not None:
        if NORMALIZE_MEAN is None or NORMALIZE_STD is None:
            raise ValueError("Set both NORMALIZE_MEAN and NORMALIZE_STD, or set both to None.")
        mean = torch.tensor(NORMALIZE_MEAN, dtype=torch.float32).view(1, -1, 1, 1)
        std = torch.tensor(NORMALIZE_STD, dtype=torch.float32).view(1, -1, 1, 1)
        if mean.shape[1] != tensor.shape[1] or std.shape[1] != tensor.shape[1]:
            raise ValueError("Normalization channels do not match IMAGE_MODE / input image channels.")
        if torch.any(std == 0):
            raise ValueError("NORMALIZE_STD values must not be zero.")
        tensor = (tensor - mean) / std

    return tensor.to(DEVICE)


def _extract_tensor(model_output) -> torch.Tensor:
    """Support common classifier outputs; ask for a custom adapter if needed."""
    if isinstance(model_output, dict):
        # Common output dictionary names from custom/vision models.
        for key in ("logits", "out", "predictions", "prediction"):
            if key in model_output:
                model_output = model_output[key]
                break
        else:
            raise ValueError("Model returned a dictionary with an unknown output key.")
    elif isinstance(model_output, (tuple, list)):
        if len(model_output) != 1:
            raise ValueError("Model returned multiple outputs; its output format needs a custom adapter.")
        model_output = model_output[0]

    if not torch.is_tensor(model_output):
        raise TypeError("Model output is not a Tensor. Check the model's forward() return value.")

    return model_output.detach().float().cpu()


def interpret_prediction(model_output):
    """Convert model outputs into a predicted label and display score.

    OUTPUT_MODE determines whether the model returns logits or probabilities.
    Do not guess this: confirm the final layer and training loss with your friend.
    """
    if OUTPUT_MODE is None:
        raise ValueError(
            "OUTPUT_MODE is not configured. Ask your friend whether the model returns "
            "one binary logit, two class logits, probabilities, or multiclass scores."
        )

    values = _extract_tensor(model_output)

    if OUTPUT_MODE in ("binary_logit", "binary_probability"):
        # One scalar represents the positive class, assumed to be CLASS_NAMES[1].
        values = values.reshape(-1)
        if values.numel() != 1:
            raise ValueError("This OUTPUT_MODE expects exactly one output value per image.")

        score = float(values[0].item())
        if OUTPUT_MODE == "binary_logit":
            positive_probability = float(torch.sigmoid(values[0]).item())
        else:
            positive_probability = score
            if not 0.0 <= positive_probability <= 1.0:
                raise ValueError("Binary probability output must be between 0 and 1.")

        # The single output is interpreted as probability of CLASS_NAMES[1].
        probabilities = [1.0 - positive_probability, positive_probability]

    elif OUTPUT_MODE in (
        "two_class_logits", "two_class_probabilities",
        "multiclass_logits", "multiclass_probabilities",
    ):
        # Standard classification shape is [batch, number_of_classes].
        # Remove the batch dimension for this one-image app.
        if values.ndim == 2 and values.shape[0] == 1:
            values = values[0]
        if values.ndim != 1:
            raise ValueError(
                f"Expected one score per class; got output shape {tuple(values.shape)}. "
                "This may be a segmentation model or require custom output handling."
            )
        if values.numel() != len(CLASS_NAMES):
            raise ValueError(
                f"Model returned {values.numel()} scores, but CLASS_NAMES has "
                f"{len(CLASS_NAMES)} labels. Match the label list to the training classes."
            )

        if OUTPUT_MODE.endswith("_logits"):
            probabilities = F.softmax(values, dim=0).tolist()
        else:
            probabilities = values.tolist()
            if any(p < 0.0 or p > 1.0 for p in probabilities):
                raise ValueError("Probability outputs must be between 0 and 1.")
            if not np.isclose(sum(probabilities), 1.0, atol=0.05):
                raise ValueError(
                    "Class probabilities do not sum approximately to 1. Confirm OUTPUT_MODE "
                    "or adapt this code to the model's output layer."
                )
    else:
        raise ValueError(
            "Unsupported OUTPUT_MODE. See the configuration comments at the top of app.py."
        )

    if len(CLASS_NAMES) != len(probabilities):
        raise ValueError("CLASS_NAMES must have one label for each interpreted class score.")

    predicted_index = int(np.argmax(probabilities))
    label = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index])
    return label, confidence, probabilities


def normalize_label(label: str) -> str:
    """Normalize label text so 'No_Tumor' and 'no tumor' match."""
    return "".join(character for character in label.lower() if character.isalnum())


# ============================================================
# 3. Header / branding
# ============================================================
st.markdown('<div class="brand">🧠 BrainCare<span>.AI</span></div>', unsafe_allow_html=True)
st.markdown('<div class="eyebrow">AI research interface · MRI imaging</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="hero">
        <div style="font-size:12px;letter-spacing:2px;color:#85e0d4;font-weight:700;">
            BRAIN MRI CLASSIFICATION
        </div>
        <h1>A clearer view.<br>A smarter first step.</h1>
        <p>Upload an MRI image to run the connected PyTorch model. For education and research only — not clinical diagnosis.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 4. Sidebar — workflow and configuration status
# ============================================================
with st.sidebar:
    st.markdown("## 🧠 BrainCare AI")
    st.caption("MRI classification workspace")
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("1. Upload an MRI image.\n2. Preview the image.\n3. Run the trained model.\n4. Review the predicted class.")
    st.markdown("---")
    st.markdown("### Configuration")
    if MODEL_PATH.exists():
        st.success(f"Model file found: {MODEL_PATH.name}")
    else:
        st.warning(f"Waiting for model file: {MODEL_PATH.name}")
    st.caption(f"Device: {DEVICE}")
    if MODEL_FORMAT not in ("torchscript", "state_dict"):
        st.warning("Set MODEL_FORMAT near the top of app.py.")
    st.info("Best-effort mode: class mapping and original preprocessing are unknown. Results are experimental, not tumor/no-tumor diagnoses.")

# ============================================================
# 5. Quick information cards
# ============================================================
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="metric"><div class="metric-label">Analysis type</div><div class="metric-value">Brain MRI</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric"><div class="metric-label">Input format</div><div class="metric-value">JPG / PNG</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric"><div class="metric-label">AI framework</div><div class="metric-value">PyTorch</div></div>', unsafe_allow_html=True)
st.write("")

# ============================================================
# 6. Image uploader + prediction result area
# ============================================================
left_col, right_col = st.columns([1.05, 0.95], gap="large")

with left_col:
    st.markdown("### 01 · Upload MRI")
    st.markdown(
        '<div class="panel"><b>Image input</b><br><span style="color:#708096;font-size:13px;">Choose an MRI image in JPG, JPEG, or PNG format.</span></div>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Choose MRI image",
        type=["jpg", "jpeg", "png"],
        help="Use a supported image for this educational demo.",
    )
    image = None
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded MRI image", use_container_width=True)
            st.caption(f"Image dimensions: {image.width} × {image.height} pixels")
        except Exception:
            st.error("Could not read this image. Please upload a valid image.")
    else:
        st.markdown(
            '<div class="panel" style="text-align:center;padding:35px 15px;"><div style="font-size:64px;">🧠</div><h4 style="color:#172d50;">Your MRI goes here</h4><p style="color:#708096;">Upload an image to begin the demonstration.</p></div>',
            unsafe_allow_html=True,
        )

with right_col:
    st.markdown("### 02 · Analysis result")
    st.markdown(
        '<div class="panel"><b>Prediction workspace</b><br><span style="color:#708096;font-size:13px;">The model result will appear here after successful inference.</span></div>',
        unsafe_allow_html=True,
    )

    config_ready = MODEL_FORMAT in ("torchscript", "state_dict") and OUTPUT_MODE is not None
    analyze_button = st.button(
        "🔍  Analyze MRI image",
        type="primary",
        use_container_width=True,
        disabled=(image is None or not config_ready),
    )

    if not config_ready:
        st.info("Model configuration is incomplete. Check MODEL_FORMAT and OUTPUT_MODE near the top of app.py.")

    if analyze_button and image is not None:
        try:
            # Load the real model; no simulated/fallback result is shown.
            model = load_model()
            model_input = preprocess_image(image)

            # Inference mode disables autograd and reduces unnecessary memory usage.
            with torch.inference_mode():
                raw_output = model(model_input)

            label, confidence, class_probabilities = interpret_prediction(raw_output)

            # Mascot is a UI decoration. It is not itself a medical assessment.
            normalized = normalize_label(label)
            normalized_no_tumor = {normalize_label(x) for x in NO_TUMOR_LABELS}
            normalized_tumor = {normalize_label(x) for x in TUMOR_LABELS}
            if normalized in normalized_no_tumor:
                st.markdown('<div class="mascot">😊🧠</div>', unsafe_allow_html=True)
                st.success(f"Model predicted class: {label}")
            elif normalized in normalized_tumor:
                st.markdown('<div class="mascot">🥺🧠</div>', unsafe_allow_html=True)
                st.warning(f"Model predicted class: {label}")
            else:
                # The checkpoint does not encode the label names/order, so
                # don't falsely claim Class 0 or Class 1 means tumor/no tumor.
                st.markdown('<div class="mascot">🤔🧠</div>', unsafe_allow_html=True)
                st.info(f"Model predicted {label}. The class meaning is unknown from the weights alone.")

            st.markdown(f"**Highest normalized class score (not diagnostic confidence):** {confidence * 100:.2f}%")
            st.progress(min(max(confidence, 0.0), 1.0))

            # Display each score when the model has multiple classes.
            if len(CLASS_NAMES) > 2 or OUTPUT_MODE in ("two_class_logits", "two_class_probabilities"):
                st.markdown("**Class scores**")
                for class_name, score in zip(CLASS_NAMES, class_probabilities):
                    st.write(f"{class_name}: {score * 100:.2f}%")

            st.caption(
                "The two scores are treated as logits as a best-effort assumption. The class labels and "
                "image preprocessing have not been verified; this output cannot confirm or rule out a tumor. "
                "Do not use it for medical decisions."
            )

        except Exception as error:
            st.error(
                "Analysis could not run. Check the model file format, model architecture, image "
                "preprocessing, output configuration, and installed dependencies."
            )
            with st.expander("Technical error details"):
                st.code(str(error))

# ============================================================
# 7. Safety notice
# ============================================================
st.markdown("---")
st.warning(
    "Medical safety notice: This educational demonstration is not a medical device and cannot "
    "diagnose, rule out, or confirm a brain tumor. Do not use it for medical decisions. MRI images "
    "must be interpreted by a qualified healthcare professional. A reassuring prediction does not "
    "mean an image is medically normal."
)
st.markdown(
    '<div class="footer">BrainCare AI · Educational machine learning demonstration<br>Protect patient privacy when uploading or sharing medical images.</div>',
    unsafe_allow_html=True,
)
