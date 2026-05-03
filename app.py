# ============================================================
# Face Mask Detection — Streamlit App
# ============================================================
# Run with:  streamlit run app.py
# ============================================================

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# ─────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────
# CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin:0; padding:0; }

html, body, [data-testid="stAppViewContainer"], .stApp {
    font-family: 'Poppins', sans-serif !important;
    background: #f0f2f6 !important;
}

/* Hide all Streamlit chrome */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
.stDeployButton,
section[data-testid="stSidebar"] { display: none !important; }

.block-container {
    max-width: 860px !important;
    padding: 0 1rem 3rem !important;
}

/* ── Blue gradient top banner ── */
.top-banner {
    background: linear-gradient(135deg, #1a73e8 0%, #1E88E5 60%, #29b6f6 100%);
    border-radius: 0 0 18px 18px;
    padding: 1.8rem 1.5rem 1.5rem;
    text-align: center;
    margin: 0 -1rem 1.5rem;
    box-shadow: 0 4px 18px rgba(26,115,232,0.35);
}
.banner-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: #fff;
    margin-bottom: 0.3rem;
}
.banner-sub {
    font-size: 0.88rem;
    color: rgba(255,255,255,0.88);
    font-weight: 300;
}

/* ── Upload label ── */
.upload-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: #444;
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    gap: 5px;
}

/* ── File uploader override ── */
[data-testid="stFileUploader"] {
    background: #fff !important;
    border: 1.5px solid #dde3f0 !important;
    border-radius: 10px !important;
    padding: 0.9rem 1rem !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    display: none !important;
}
[data-testid="stBaseButton-secondary"] {
    background: #fff !important;
    border: 1.5px solid #bcc5e4 !important;
    color: #1a73e8 !important;
    font-family: 'Poppins', sans-serif !important;
    font-weight: 500 !important;
    border-radius: 7px !important;
    padding: 0.4rem 1.1rem !important;
    font-size: 0.85rem !important;
}

/* ── Left panel: uploaded image ── */
.img-panel {
    background: #fff;
    border-radius: 14px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    overflow: hidden;
    text-align: center;
}
.img-caption {
    font-size: 0.76rem;
    color: #9aa5b4;
    padding: 0.5rem;
    font-weight: 400;
}

/* ── Result boxes ── */
.result-mask {
    background: #fff;
    border: 2.5px solid #38a169;
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
    font-size: 1.15rem;
    font-weight: 700;
    color: #276749;
    letter-spacing: 0.01em;
}
.result-nomask {
    background: #fff;
    border: 2.5px solid #e53e3e;
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
    font-size: 1.15rem;
    font-weight: 700;
    color: #9b2c2c;
    letter-spacing: 0.01em;
}

/* ── Confidence section ── */
.conf-section {
    background: #fff;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    padding: 14px 16px;
    margin-top: 12px;
}
.conf-heading {
    font-size: 0.9rem;
    font-weight: 600;
    color: #2d3748;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.conf-bar-track {
    background: #e2e8f0;
    border-radius: 50px;
    height: 28px;
    overflow: hidden;
    margin-bottom: 10px;
}
.conf-bar-inner {
    background: linear-gradient(90deg, #1a73e8, #29b6f6);
    height: 100%;
    border-radius: 50px;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding-right: 12px;
    font-size: 0.82rem;
    font-weight: 700;
    color: #fff;
    min-width: 60px;
}
.info-line {
    font-size: 0.82rem;
    color: #4a5568;
    margin-bottom: 4px;
    line-height: 1.5;
}
.info-line b { font-weight: 600; color: #1a202c; }

/* ── Confidence badges ── */
.badge-high {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: #f0fff4;
    border: 1px solid #9ae6b4;
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 0.83rem;
    font-weight: 600;
    color: #276749;
    margin-top: 4px;
}
.badge-mid {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: #ebf8ff;
    border: 1px solid #90cdf4;
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 0.83rem;
    font-weight: 600;
    color: #2b6cb0;
    margin-top: 4px;
}
.badge-low {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: #fffaf0;
    border: 1px solid #fbd38d;
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 0.83rem;
    font-weight: 600;
    color: #c05621;
    margin-top: 4px;
}

/* ── Footer ── */
.footer {
    text-align: center;
    margin-top: 2.2rem;
    font-size: 0.8rem;
    color: #a0aec0;
}

/* ── Placeholder ── */
.placeholder {
    background: #fff;
    border: 2px dashed #e2e8f0;
    border-radius: 14px;
    text-align: center;
    padding: 3.5rem 2rem;
    margin-top: 1rem;
    color: #cbd5e0;
}
.ph-icon { font-size: 2.8rem; margin-bottom: 0.7rem; }
.ph-text  { font-size: 0.88rem; font-weight: 500; }

/* Fade in */
@keyframes fadeUp {
    from { opacity:0; transform:translateY(10px); }
    to   { opacity:1; transform:translateY(0); }
}
.anim { animation: fadeUp 0.35s ease both; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────
import os
import gdown
import tensorflow as tf
import streamlit as st

MODEL_PATH = "face_mask_model.h5"
FILE_ID = "1MfNmbEMB0NFTfqa0sysK0GnSW68jPKbD"   # 👈 add your Google Drive file ID

@st.cache_resource
def load_model():
    # Download model if not present
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

    # Load model
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ─────────────────────────────────────────
# Top Banner
# ─────────────────────────────────────────
st.markdown("""
<div class="top-banner">
    <div class="banner-title">😷 Face Mask Detection</div>
    <div class="banner-sub">Upload an image to detect mask usage</div>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model not found. Run `python train_model.py` first to generate `face_mask_model.h5`.")
    st.stop()

# ─────────────────────────────────────────
# Preprocessing — must match train_model.py
# ✅ Updated to 224×224 + rescale 1/255
# ─────────────────────────────────────────
IMG_SIZE = (224, 224)   # ✅ Updated from 128 → 224

def preprocess_image(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0  # Normalize [0, 1]
    return np.expand_dims(arr, axis=0)              # (1, 224, 224, 3)

# ─────────────────────────────────────────
# Upload
# ─────────────────────────────────────────
st.markdown('<div class="upload-label">📤 Upload Image</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "200MB per file • JPG, PNG, JPEG",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)

# ─────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    with st.spinner("Detecting..."):
        arr = preprocess_image(image)
        raw = float(model.predict(arr)[0][0])

    # Label mapping:
    # Keras alphabetical: "with_mask"=0 → raw<0.5 ✅ | "without_mask"=1 → raw≥0.5 ❌
    is_mask  = raw < 0.5
    conf     = (1.0 - raw) if is_mask else raw
    conf_pct = conf * 100
    bar_w    = max(int(conf_pct), 8)

    if is_mask:
        result_box = '<div class="result-mask anim">✅ &nbsp; Wearing Mask</div>'
        pred_label = "With Mask"
    else:
        result_box = '<div class="result-nomask anim">✖ &nbsp; No Mask</div>'
        pred_label = "Without Mask"

    if conf_pct >= 80:
        badge = '<div class="badge-high">🔥 &nbsp; High Confidence</div>'
    elif conf_pct >= 55:
        badge = '<div class="badge-mid">✅ &nbsp; Medium Confidence</div>'
    else:
        badge = '<div class="badge-low">⚠️ &nbsp; Low Confidence</div>'

    col1, col2 = st.columns([1, 1], gap="small")

    # Left — Image
    with col1:
        st.markdown('<div class="img-panel anim">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('<div class="img-caption">Uploaded image</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Right — Result
    with col2:
        st.markdown(result_box, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="conf-section anim">
            <div class="conf-heading">📊 &nbsp; Confidence Level</div>
            <div class="conf-bar-track">
                <div class="conf-bar-inner" style="width:{bar_w}%;">
                    {conf_pct:.1f}%
                </div>
            </div>
            <div class="info-line"><b>Prediction:</b> {pred_label}</div>
            <div class="info-line"><b>Confidence:</b> {conf_pct:.2f}%</div>
            {badge}
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="placeholder">
        <div class="ph-icon">🖼️</div>
        <div class="ph-text">Upload a face image above to get started</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# Footer
# ─────────────────────────────────────────
st.markdown('<div class="footer">Made with ❤️ using Streamlit</div>', unsafe_allow_html=True)