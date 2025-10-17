import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from tensorflow.keras.models import load_model
import numpy as np

# ==============================
# 1. Konfigurasi Dasar
# ==============================
st.set_page_config(page_title="Multi-Model Image Classifier", layout="wide")

st.title("📸 Dashboard Klasifikasi Gambar dengan Dua Model")
st.write("Aplikasi ini memungkinkan Anda memilih model untuk melakukan klasifikasi gambar secara otomatis.")

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
uploaded_file = st.file_uploader("Unggah gambar untuk diklasifikasi", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

# ==============================
# 4. Load Model
# ==============================
@st.cache_resource
def load_keras_model():
    return load_model("klasifikasi 8 kelas.h5")

@st.cache_resource
def load_torch_model():
    from ultralytics import YOLO

    # Load model YOLO (deteksi gender)
    model = YOLO("deteksi gender.pt")
    return model


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


def predict_torch(image):
    model = load_torch_model()

    # Jalankan prediksi
    results = model.predict(image, imgsz=224, conf=0.5, verbose=False)

    # Kalau model deteksi (punya bounding box)
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        # Ambil kelas dan confidence tertinggi
        boxes = results[0].boxes
        cls_idx = int(boxes.cls[0].item())
        conf = float(boxes.conf[0].item())
        label = f"{results[0].names[cls_idx]} ({conf:.2f})"

        # Gambar hasil deteksi
        img_plot = results[0].plot()
        st.image(img_plot, caption="Hasil Deteksi YOLO")

        return label, [conf]
    
    # Kalau model klasifikasi (punya .probs)
    elif results[0].probs is not None:
        probs = results[0].probs.data.cpu().numpy()
        label_idx = results[0].probs.top1
        label = results[0].names[label_idx]
        return label, probs

    else:
        return "Tidak ada deteksi yang ditemukan", [0.0]


# ==============================
# 6. Tombol Prediksi
# ==============================
if uploaded_file:
    if st.button("🔍 Jalankan Prediksi"):
        with st.spinner("Model sedang memproses gambar..."):
            if model_choice == "Klasifikasi 8 Kelas":
                label, prob = predict_keras(image)
            elif model_choice == "Deteksi Gender":
                label, prob = predict_torch(image)

        st.success("Prediksi berhasil!")
        st.subheader("🧾 Hasil Prediksi:")
        st.write(f"**Label:** {label}")
        st.bar_chart(prob)

# ==============================
# 7. Catatan
# ==============================
st.markdown("""
---
**Petunjuk Penggunaan:**
1. Pilih model dari sidebar.
2. Unggah gambar (format JPG/PNG).
3. Klik **Jalankan Prediksi**.
4. Lihat hasil klasifikasi dan grafik probabilitas.
""")