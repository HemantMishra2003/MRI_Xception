# ================= ALL IMPORTANT LIBRARIES =================
import os
import streamlit as st
import tensorflow as tf
import numpy as np
from datetime import datetime
from PIL import Image

from tensorflow.keras.preprocessing import image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import qrcode


# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="🧠 MRI AI Cancer Detector Model",
    layout="centered"
)

#  QR DATA
QR_DATA = "https://github.com/HemantMishra2003"


#   UI
st.markdown("""
<style>
label, .stTextInput label {
    margin-bottom: -25px !important;
}
.stTextInput {
    margin-top: -10px !important;
}
.stTextInput input {
    font-size: 17px !important;
    font-weight: 600 !important;
}
@keyframes blink {
  0% {opacity: 1;}
  50% {opacity: 0.3;}
  100% {opacity: 1;}
}
.detect-box {
  border: 2px solid red;
  padding: 18px;
  border-radius: 10px;
  text-align: center;
  animation: blink 1s infinite;
  margin-top: 20px;
}
.detect-title {
  color: red;
  font-size: 26px;
  font-weight: 800;
}
.detect-conf {
  color: red;
  font-size: 18px;
  margin-top: 6px;
  font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


#     ASSETS
CLASS_IMAGE_MAP = {
    "Glioma": "assets/Te-gl_0224.jpg",
    "Meningioma": "assets/Te-me_0010.jpg",
    "Pituitary": "assets/Te-pi_0025.jpg",
    "No Tumor": "assets/Te-no_0114.jpg",
}
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

CANCER_LABEL_MAP = {
    "Glioma": "Glioma Cancer Detected",
    "Meningioma": "Meningioma Cancer Detected",
    "Pituitary": "Pituitary Gland Cancer Detected",
    "No Tumor": "No Tumor Detected"
}

PDF_CANCER_LABEL_MAP = {
    "Glioma": "GLIOMA BRAIN CANCER DETECTED",
    "Meningioma": "MENINGIOMA BRAIN CANCER DETECTED",
    "Pituitary": "PITUITARY GLAND CANCER DETECTED",
    "No Tumor": "NO TUMOR DETECTED"
}


# ================= HEADER =================
st.title("🧠 MRI AI Cancer Detector Model")
st.markdown(
    "<p style='color:white; margin-top:-10px; margin-left:75px; "
    "font-size:17px; font-weight:650;'>"
    "(Xception | Transfer Learning)</p>",
    unsafe_allow_html=True
)

# ================= INPUT =================
st.markdown("<p style='font-size:20px; font-weight:600;'>Patient Name</p>", unsafe_allow_html=True)
name = st.text_input("", placeholder="Enter Patient Name")

st.markdown("<p style='font-size:20px; font-weight:600;'>Patient Age</p>", unsafe_allow_html=True)
age = st.number_input("", min_value=0, max_value=120, step=1)

st.markdown(
    "<p style='font-size:20px; font-weight:600;'>Upload MRI Image</p>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "",
    type=["jpg", "png", "jpeg"]
)



# ================= MODEL LOAD (WEIGHTS BASED – FINAL SAFE) =================
@st.cache_resource
def load_model():
    img_shape = (299, 299, 3)

    base_model = tf.keras.applications.Xception(
        include_top=False,
        weights="imagenet",
        input_shape=img_shape
    )

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(4, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adamax(0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    from huggingface_hub import hf_hub_download
    weights_path = hf_hub_download(
        repo_id="HemantMishraDeepak/newXception",
        filename="xception_brain_tumor_weights.weights.h5"
    )

    model.load_weights(weights_path)
    return model


model = load_model()
st.success("Your Model loaded successfully")


# ================= PREDICTION =================
def predict_mri(uploaded_file):
    img = image.load_img(uploaded_file, target_size=(299, 299))
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr, verbose=0)[0]
    probs = {CLASS_NAMES[i]: round(float(preds[i]) * 100, 2) for i in range(4)}
    detected = CLASS_NAMES[np.argmax(preds)]
    return detected, probs


# ================= PDF GENERATOR  =================
def generate_pdf(img_path, name, age, detected, probs):
    pdf = "MRI_AI_Report.pdf"

    # ---------- QR ----------
    qr = qrcode.make(QR_DATA)
    qr_path = "temp_qr.png"
    qr.save(qr_path)

    c = canvas.Canvas(pdf, pagesize=A4)
    w, h = A4
    y = h - 40

    # ================= HEADER =================
    c.setFont("Helvetica-Bold", 19)
    c.drawCentredString(
        w/2, y,
        "SHRI HEMKUND INSTITUTE OF RADIOLOGY & IMAGING"
    )
    y -= 28

    c.setFont("Helvetica-Bold", 15)
    c.drawCentredString(w/2, y, "MRI AI Diagnostic Report")
    y -= 16

    c.setFont("Helvetica-Bold", 13)
    c.drawCentredString(
        w/2, y,
        "(Fine-Tuned Xception | Transfer Learning)"
    )
    y -= 25

    # QR top-right
    c.drawImage(qr_path, w-110, h-120, 70, 70)

    # ================= PATIENT INFO =================
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y, f"Name : {name}")
    y -= 14
    c.drawString(50, y, f"Age  : {age}")
    y -= 14
    c.drawString(
        50, y,
        f"Date : {datetime.now().strftime('%d-%m-%Y')}"
    )
    y -= 25

    # ================= MAIN RESULT =================
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(
        w/2, y,
        PDF_CANCER_LABEL_MAP[detected]
    )
    y -= 18

    c.setFont("Helvetica-Bold", 13)
    c.drawCentredString(
        w/2, y,
        f"Confidence: {probs[detected]:.2f}%"
    )
    y -= 25

    # TWO IMAGES
    img_w, img_h = 200, 160
    left_x = 50
    right_x = w/2 + 20

    # Uploaded MRI (left)
    c.drawImage(img_path, left_x, y-img_h, img_w, img_h)

    # Reference MRI (right)
    c.drawImage(
        CLASS_IMAGE_MAP["No Tumor"],
        right_x, y-img_h, img_w, img_h
    )

    # Labels
    c.setFont("Helvetica-Bold", 11)
    c.drawCentredString(
        left_x + img_w/2,
        y - img_h - 14,
        f"Uploaded MRI : {detected}"
    )
    c.drawCentredString(
        right_x + img_w/2,
        y - img_h - 14,
        "Reference MRI"
    )

    y -= img_h + 40

    # ================= THREE IMAGES (BOTTOM) =================
    small_w, small_h = 130, 130
    x_pos = [40, 200, 360]

    for i, cls in enumerate(["Glioma", "Meningioma", "Pituitary"]):
        c.drawImage(
            CLASS_IMAGE_MAP[cls],
            x_pos[i], y-small_h, small_w, small_h
        )
        c.drawCentredString(
            x_pos[i] + small_w/2,
            y - small_h - 14,
            cls
        )
        c.drawCentredString(
            x_pos[i] + small_w/2,
            y - small_h - 28,
            f"Confidence: {probs[cls]:.2f}%"
        )

    y -= small_h + 55

    # ================= FOOTER =================
    c.setFont("Helvetica-Bold", 13)
    c.drawCentredString(
        w/2, y,
        "Scan my QR code to see my other Projects"
    )
    y -= 20

    c.setFont("Helvetica-Bold", 15)
    c.drawCentredString(
        w/2, y,
        "Developed under the Guidance of Dr. Vishwas Mishra Sir ( Rolls-Royce )"
    )

    c.save()
    os.remove(qr_path)
    return pdf



# ================= MAIN FLOW =================
if uploaded_file and name and age:
    with open("uploaded_temp.jpg", "wb") as f:
        f.write(uploaded_file.read())

    st.image(uploaded_file, caption="Uploaded MRI", width=320)

    detected, probs = predict_mri(uploaded_file)

    if detected != "No Tumor":
        st.markdown(
            f"""
            <div class="detect-box">
                <div class="detect-title">{CANCER_LABEL_MAP[detected]}</div>
                <div class="detect-conf">Confidence: {probs[detected]}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.success(f"No Tumor Detected (Confidence: {probs['No Tumor']}%)")

    st.subheader("Reference MRI Images with Confidence")
    cols = st.columns(4)
    for col, cls in zip(cols, CLASS_IMAGE_MAP.keys()):
        with col:
            st.image(CLASS_IMAGE_MAP[cls], width=200)
            st.markdown(f"<b>{cls}</b>", unsafe_allow_html=True)
            st.markdown(f"Confidence: {probs[cls]}%")

    if st.button("Generate MRI PDF Report"):
        pdf = generate_pdf("uploaded_temp.jpg", name, age, detected, probs)
        with open(pdf, "rb") as f:
            st.download_button("Download MRI Report PDF", f, file_name=pdf)

else:
    st.info("👆 Please enter patient details and upload an MRI image.")

