# BrainCare AI — inspected PyTorch checkpoint

This project contains the uploaded `tumor_cnn.pth` checkpoint plus a Streamlit app.

## What was verified

The uploaded file successfully loads with `torch.load(..., map_location="cpu", weights_only=True)` as a PyTorch state dictionary. Its learned tensor keys/shapes indicate:

- `features.0.weight`: `(16, 3, 3, 3)` — first convolution accepts 3 channels and produces 16 channels.
- `features.3.weight`: `(32, 16, 3, 3)` — second convolution produces 32 channels.
- `classifier.1.weight`: `(64, 2048)` — first dense layer takes 2048 features and produces 64.
- `classifier.3.weight`: `(2, 64)` — final dense layer produces 2 output scores.

This is a weights-only state dictionary, not a self-contained model. The supplied checkpoint does not identify class names/order or the original image preprocessing. `model_definition.py` is a best-effort reconstruction from the layer names/shapes; the original training notebook/model class is still the authoritative source.

## Before enabling prediction

Open `app.py` and confirm these settings with the model creator:

1. `OUTPUT_MODE`: use `"two_class_logits"` only if `forward()` returns two raw logits; use `"two_class_probabilities"` if it returns probabilities.
2. `CLASS_NAMES`: set these in the exact training order. Do not assume which index means “tumor”.
3. `IMAGE_SIZE`, `IMAGE_MODE`, and normalization: use the exact settings from training.
4. Confirm the non-parameter layers in `model_definition.py` against the original class. The weights load into this inferred structure, but matching tensor shapes alone cannot prove every operation is identical.

Prediction is disabled by default (`OUTPUT_MODE = None`) to avoid inventing a medical result or class mapping.

## Run in Colab

1. Upload/extract this project in your Colab session.
2. Install dependencies with `pip install -r requirements.txt`.
3. Confirm configuration above with the model creator.
4. To test, use `streamlit run app.py` locally, or package the project and deploy it through GitHub + Streamlit Community Cloud.

## Safety

This is an educational prototype, not a clinically validated diagnostic system. Never use it to diagnose or rule out a tumor. MRI studies need review by a qualified medical professional. Do not upload identifiable patient images to a public repository or app.
