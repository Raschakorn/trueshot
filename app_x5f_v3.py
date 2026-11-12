# TrueShot - Prototype for Distinguishing AI-Generated Images from Photographs
# Methods: Metadata analysis + Error Level Analysis + Frequency domain analysis + Noise residual
# Status: Prototype, heuristic scoring, not validated on benchmark datasets.
# Run with: streamlit run app_v3.py

import streamlit as st
from PIL import Image
import numpy as np, io, cv2

st.set_page_config(page_title="TrueShot - Prototype", page_icon="📷", layout="wide")
st.title("TrueShot - Prototype")
st.write("An experimental tool for distinguishing AI-generated images from photographs by combining metadata and pixel-level evidence. This version uses heuristic scoring and has not been validated on standard benchmark datasets. Results should be interpreted as a probability estimate and reviewed by a human.")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption=f"{uploaded.name} {img.size[0]}x{img.size[1]}", use_container_width=True)

    # --- 1. Metadata and Provenance Analysis ---
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw = buf.getvalue().lower()
    has_c2pa_string = b"c2pa" in raw or b"jumbf" in raw
    exif = img.getexif()
    has_camera_exif = any("canon" in str(v).lower() or "nikon" in str(v).lower() or "sony" in str(v).lower() or "iphone" in str(v).lower() for v in exif.values()) if exif else False
    info_text = str(img.info).lower()
    ai_keywords = [k for k in ["stable diffusion", "midjourney", "dall", "sdxl", "comfyui", "firefly", "imagen", "synthid"] if k in info_text]
    score_prov = 50
    reasons = []
    if ai_keywords:
        score_prov += 35
        reasons.append(f"AI-related keyword found in file metadata: {ai_keywords}")
    if has_c2pa_string:
        score_prov -= 20
        reasons.append("String 'C2PA' or 'JUMBF' found in file. Note: This is a string search only. Cryptographic signature verification is not performed.")
    if has_camera_exif:
        score_prov -= 25
        reasons.append("Camera make found in EXIF metadata.")
    elif not exif:
        score_prov += 10
        reasons.append("No EXIF metadata present.")
    if b"synthid" in raw:
        score_prov += 30
        reasons.append("String 'SynthID' found in file. Note: Watermark decoding is not performed.")

    # --- 2. Pixel-Level Forensic Analysis ---
    # ELA + FFT + Laplacian + JPEG block artifact + Chroma analysis
    # All scores are heuristic. No accuracy claim is made in code.
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    lap_var = float(np.var(cv2.Laplacian(gray, cv2.CV_32F)))
    score_for = 50
    if lap_var < 20:
        score_for += 15
        reasons.append(f"Low Laplacian variance ({lap_var:.1f}). Image is unusually smooth.")
    elif lap_var < 50:
        score_for += 7
        reasons.append(f"Moderately low Laplacian variance ({lap_var:.1f}).")
    elif lap_var > 250:
        score_for -= 12
        reasons.append(f"High Laplacian variance ({lap_var:.1f}). Consistent with natural sensor noise.")
    # Frequency domain analysis using 2D Fourier Transform
    f = np.fft.fft2(gray)
    mag = np.abs(np.fft.fftshift(f))
    h, w = mag.shape
    ratio = float(mag[h//4:3*h//4, w//4:3*w//4].mean() / (mag.mean() + 1e-6))
    if ratio > 10:
        score_for += 8
        reasons.append("Strong central concentration in Fourier spectrum.")
    elif ratio > 8:
        score_for += 4
        reasons.append("Moderate central concentration in Fourier spectrum.")
    # JPEG 8x8 block artifact check
    try:
        arr_u8 = np.asarray(img.convert("L"))
        horiz_bound = np.abs(arr_u8[:, 7::8].astype(int) - arr_u8[:, 8::8].astype(int)).mean() if arr_u8.shape[1] > 9 else 0
        horiz_inside = np.abs(arr_u8[:, 3::8].astype(int) - arr_u8[:, 4::8].astype(int)).mean() if arr_u8.shape[1] > 9 else 0
        block_diff = float(horiz_bound - horiz_inside)
        if block_diff < 0.5:
            score_for += 6
            reasons.append(f"Weak JPEG block artifact (diff {block_diff:.2f}). May indicate regeneration.")
        elif block_diff > 2.0:
            score_for -= 6
            reasons.append(f"Clear JPEG block artifact (diff {block_diff:.2f}). Consistent with direct camera JPEG.")
    except Exception:
        pass
    # Chroma variance check
    try:
        ycrcb = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2YCrCb)
        cr_var = float(np.var(ycrcb[:, :, 1].astype(np.float32)))
        cb_var = float(np.var(ycrcb[:, :, 2].astype(np.float32)))
        chroma_var = (cr_var + cb_var) / 2.0
        if chroma_var < 80:
            score_for += 5
            reasons.append(f"Low chroma variance ({chroma_var:.1f}).")
    except Exception:
        pass

    # --- 3. Face Detection (Preliminary Semantic Check) ---
    # Note: Haar cascade is disabled on Streamlit Cloud because cv2.data is not
    # available in opencv-python-headless. Face check is therefore skipped
    # safely instead of crashing the app.
    score_sem = 50
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml" if hasattr(cv2, "data") else ""
        if cascade_path:
            face_cascade = cv2.CascadeClassifier(cascade_path)
            if not face_cascade.empty():
                faces = face_cascade.detectMultiScale(np.asarray(img.convert("L")), 1.1, 4)
                if len(faces) > 0:
                    reasons.append(f"{len(faces)} face(s) detected. Manual inspection of eyes, teeth, and hands is recommended.")
                else:
                    reasons.append("No clear face detected. Face-based semantic check skipped.")
            else:
                reasons.append("Face detector unavailable in this environment. Face check skipped.")
        else:
            reasons.append("Face detector unavailable in this environment. Face check skipped.")
    except Exception:
        reasons.append("Face detector unavailable in this environment. Face check skipped.")

    # --- 4. Combined Score ---
    # Weighted combination of heuristic scores. This is not an ensemble of trained models.
    score_combined = int(score_prov * 0.4 + score_for * 0.4 + score_sem * 0.2)

    # --- 5. Final Score and Interpretation ---
    final = int(score_prov * 0.30 + score_for * 0.25 + score_sem * 0.15 + score_combined * 0.30)
    final = max(0, min(100, final))

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final probability of AI generation", f"{final}%")
    c2.metric("Metadata score", f"{score_prov}%")
    c3.metric("Pixel forensic score", f"{score_for}%")
    c4.metric("Combined score", f"{score_combined}%")

    if final >= 70:
        st.error("High probability of AI generation (prototype estimate - human review required).")
    elif final >= 45:
        st.warning("Inconclusive - additional evidence required.")
    else:
        st.success("Low probability of AI generation (prototype estimate - human review required).")

    st.subheader("Evidence and Reasoning")
    for r in reasons:
        st.write("- " + r)

    st.subheader("Visual Evidence")
    colA, colB = st.columns(2)
    # Error Level Analysis (ELA)
    buf2 = io.BytesIO()
    img.save(buf2, format="JPEG", quality=90)
    buf2.seek(0)
    comp = Image.open(buf2)
    diff = np.abs(np.asarray(img).astype(int) - np.asarray(comp).astype(int)) * 10
    colA.image(Image.fromarray(np.clip(diff, 0, 255).astype(np.uint8)), caption="Error Level Analysis (ELA) - Bright regions indicate higher recompression difference.")
    # Noise residual
    arr = np.asarray(img)
    blur = cv2.GaussianBlur(arr, (0, 0), 3)
    resid = cv2.normalize(cv2.absdiff(arr, blur), None, 0, 255, cv2.NORM_MINMAX)
    colB.image(Image.fromarray(resid), caption="Noise Residual - Natural photographs typically show uniform noise distribution.")

    st.info("Limitations: This is a prototype. It has not been evaluated on GenImage, CIFAKE, or other benchmark datasets. It does not perform cryptographic C2PA verification, SynthID decoding, PRNU analysis, or Thai OCR. Scores are heuristic and should not be used as sole evidence for classification.")
