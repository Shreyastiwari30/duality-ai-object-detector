import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os
import time

# --- Page config ---
st.set_page_config(
    page_title="Object Detector - Duality AI Hackathon",
    page_icon="🚀",
    layout="centered"
)

# --- Title ---
st.title("🚀 Duality AI - Space Station Object Detector")
st.markdown("Detects objects like **cheerios** and **soup** using a custom-trained YOLOv8 model.")

# --- Load model ---
@st.cache_resource
def load_model():
    return YOLO("runs/detect/train2/weights/best.pt")

model = load_model()

# --- Image uploader ---
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    # --- Prediction ---
    with st.spinner("🔍 Detecting... please wait"):
        output_dir = "predicted"
        results = model.predict(source=image, save=True, save_txt=False, project=output_dir, name="results", exist_ok=True, conf=0.25)
        time.sleep(1)

    # --- Show result ---
    result_path = os.path.join(output_dir, "results", os.listdir(f"{output_dir}/results")[0])
    st.image(result_path, caption="✅ Detection Result", use_column_width=True)

    st.success("🎯 Detection Complete!")
    st.markdown("### 📊 Model Confidence Threshold: `0.25`\nCustomize in code if needed.")

# --- Footer ---
st.markdown("---")
st.caption("🛰️ Built for the Duality AI Hackathon | Team Code Wizards 🚀")
