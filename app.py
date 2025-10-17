import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from tensorflow.keras.models import load_model
import numpy as np
from ultralytics import YOLO

# ==============================
# 1. Konfigurasi Dasar
# ==============================
st.set_page_config(page_title="Multi-Model Image Classifier", layout="wide")

st.title("📸 Dashboard Klasifikasi Gambar dengan Dua Model")
st.write("Aplikasi ini memungkinkan Anda memilih model untuk melakukan klasifikasi gambar atau deteksi gender secara otomatis.")

# ==============================
# 2. Sidebar Pilihan Model
# ==============================
st.sidebar.header("🧠 Pilih Model yang Akan Digunakan")
model_choice = st.sidebar.selectbox(
    "Pilih Model:",
    ["Klasifikasi 8 Kelas", "Deteksi Gender"]
)

# ==============================
# 3. Upload Gambar
# ==============================
uploaded_file = st.file_uploader("Unggah gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

# ==============================
# 4. Load Model
# ==============================
@st.cache_resource
def load_keras_model():
    return load_model("/content/klasifikasi 8 kelas.h5")

@st.cache_resource
def load_yolo_model():
    return YOLO("/content/deteksi gender.pt")

# ==============================
# 5. Fungsi Prediksi
# ==============================
def predict_keras(image):
    model = load_keras_model()
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    class_idx = np.argmax(pred)
    kelas = [
        "Airplane (Kelas 1)",
        "Car (Kelas 2)",
        "Cat (Kelas 3)",
        "Dog (Kelas 4)",
        "Flower (Kelas 5)",
        "Fruit (Kelas 6)",
        "Motorbike (Kelas 7)",
        "Person (Kelas 8)"
    ]
    return kelas[class_idx], pred[0]


def predict_yolo(image):
    model = load_yolo_model()
    results = model.predict(image, imgsz=224, conf=0.5, verbose=False)

    # Jika deteksi bbox
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        cls_idx = int(boxes.cls[0].item())
        conf = float(boxes.conf[0].item())
        label = f"{results[0].names[cls_idx]} ({conf:.2f})"
        img_plot = results[0].plot()
        st.image(img_plot, caption="Hasil Deteksi YOLO")
        return label, [conf]

    elif results[0].probs is not None:
        probs = results[0].probs.data.cpu().numpy()
        label_idx = results[0].probs.top1
        label = results[0].names[label_idx]
        return label, probs
    else:
        return "Tidak ada deteksi", [0.0]

# ==============================
# 6. Tombol Prediksi
# ==============================
if uploaded_file:
    if st.button("🔍 Jalankan Prediksi"):
        with st.spinner("Model sedang memproses gambar..."):
            if model_choice == "Klasifikasi 8 Kelas":
                label, prob = predict_keras(image)
            else:
                label, prob = predict_yolo(image)
        st.success("✅ Prediksi berhasil!")
        st.subheader("🧾 Hasil Prediksi:")
        st.write(f"**Label:** {label}")
        st.bar_chart(prob)

# ==============================
# 7. Petunjuk
# ==============================
st.markdown("""
---
### 📘 Petunjuk
1. Pilih model di sidebar kiri.
2. Unggah gambar JPG/PNG.
3. Tekan **Jalankan Prediksi**.
4. Lihat hasil & grafik probabilitas.
""")
