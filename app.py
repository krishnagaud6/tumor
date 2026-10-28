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
