"""Local Streamlit GUI for AuthenticVision, with precise threshold override."""
from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageOps

from av_model import build_transforms, load_checkpoint

st.set_page_config(page_title="AuthenticVision — My Trained Model", page_icon="🔎", layout="wide")
st.title("AuthenticVision — Real vs AI Image Detector")
st.caption("Research/learning tool only. Results are estimates, not proof of image origin.")

checkpoint_default = str(Path("outputs") / "best_model.pt")
checkpoint_path = st.sidebar.text_input("Checkpoint path", value=checkpoint_default)
use_threshold_override = st.sidebar.checkbox("Override the trained threshold", value=False)
threshold_text = st.sidebar.text_input(
    "Custom threshold (0–1; scientific notation accepted)",
    value="0.0000079",
    help=(
        "Example: 0.0000079 or 7.9e-6. A custom threshold only changes the final "
        "decision rule; it does not improve model weights."
    ),
)

custom_threshold = None
if use_threshold_override:
    try:
        custom_threshold = float(threshold_text.strip())
        if not 0.0 <= custom_threshold <= 1.0:
            raise ValueError("Threshold must be between 0 and 1.")
    except ValueError:
        st.sidebar.error("Enter a valid number between 0 and 1, e.g. 7.9e-6.")
        st.stop()

if "loaded_checkpoint_path" not in st.session_state:
    st.session_state.loaded_checkpoint_path = None
if "model_bundle" not in st.session_state:
    st.session_state.model_bundle = None

if st.sidebar.button("Load trained model", type="primary"):
    path = Path(checkpoint_path).expanduser()
    if not path.exists():
        st.sidebar.error(f"Checkpoint not found: {path}")
        st.session_state.model_bundle = None
    else:
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, checkpoint = load_checkpoint(str(path), device)
            _, transform = build_transforms(int(checkpoint.get("image_size", 224)))
            st.session_state.model_bundle = (model, checkpoint, transform, device)
            st.session_state.loaded_checkpoint_path = str(path.resolve())
            st.sidebar.success("Model loaded successfully.")
        except Exception as exc:
            st.session_state.model_bundle = None
            st.sidebar.error(f"Could not load model: {exc}")

bundle = st.session_state.model_bundle
if bundle is None:
    st.info("Load the trained `best_model.pt` checkpoint using the sidebar.")
    st.code("streamlit run app.py", language="powershell")
    st.stop()

model, checkpoint, transform, device = bundle
trained_threshold = float(checkpoint.get("threshold", 0.5))
threshold = float(custom_threshold) if use_threshold_override else trained_threshold

st.caption(
    f"Device: {device} · Model: {checkpoint.get('model_name', 'convnext_tiny')} · "
    f"Decision threshold: {threshold:.10g}"
)
st.sidebar.caption(f"Loaded checkpoint: {st.session_state.loaded_checkpoint_path}")
st.sidebar.caption(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
st.sidebar.caption(f"Stored trained threshold: {trained_threshold:.10g}")

if use_threshold_override:
    st.warning(
        "CUSTOM THRESHOLD IS ON. A low value such as 7.9e-6 was selected from your "
        "provided examples to test that batch. It is not a validated general-purpose "
        "threshold and may label other real photos as AI-generated. Do not use this "
        "batch-fitted setting to claim model accuracy."
    )
if trained_threshold < 0.001:
    st.info(
        "The checkpoint's calibrated threshold is very small. Inspect full-precision "
        "AI scores in the table/CSV rather than rounded percentages alone."
    )

uploads = st.file_uploader(
    "Upload one or more images",
    type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"],
    accept_multiple_files=True,
)
if uploads:
    rows = []
    columns = st.columns(min(3, len(uploads)))
    for idx, upload in enumerate(uploads):
        start = time.perf_counter()
        try:
            image = ImageOps.exif_transpose(Image.open(io.BytesIO(upload.getvalue()))).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)
            with torch.inference_mode():
                logits = model(tensor)
                p_ai = float(torch.softmax(logits.float(), dim=1)[0, 1].item())
            verdict = "Likely AI-generated" if p_ai >= threshold else "Likely real"
            elapsed = time.perf_counter() - start
            with columns[idx % len(columns)]:
                st.image(image, caption=upload.name, use_container_width=True)
                st.metric("AI score", f"{p_ai * 100:.6f}%", help=f"Full-precision AI score: {p_ai:.12g}")
                st.write(f"**Verdict:** {verdict}")
                st.caption(f"Inference time: {elapsed:.2f} s")
            rows.append({
                "filename": upload.name,
                "probability_ai": f"{p_ai:.12g}",
                "probability_real": f"{1.0 - p_ai:.12g}",
                "ai_score_percent": f"{p_ai * 100.0:.10g}",
                "threshold": f"{threshold:.12g}",
                "verdict": verdict,
                "inference_seconds": round(elapsed, 4),
            })
        except Exception as exc:
            st.error(f"Could not analyse {upload.name}: {exc}")
    if rows:
        st.subheader("Results table")
        frame = pd.DataFrame(rows)
        st.dataframe(frame, use_container_width=True, hide_index=True)
        st.download_button(
            "Download CSV results",
            frame.to_csv(index=False).encode("utf-8"),
            file_name="authenticvision_predictions.csv",
            mime="text/csv",
        )
        st.warning(
            "A high score is not conclusive proof. Unseen generators, edits, and compression "
            "can cause misclassification. A threshold change does not retrain the model."
        )
