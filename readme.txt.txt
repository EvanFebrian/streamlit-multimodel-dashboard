Proyek: Multi-Model Image Classifier and Gender Detector

Deskripsi:
Aplikasi ini adalah dashboard Streamlit yang menggabungkan dua model:
1. Model Klasifikasi (.h5) untuk mengklasifikasikan gambar menjadi 8 kelas (Airplane, Car, Cat, Dog, Flower, Fruit, Motorbike, Person)
2. Model Deteksi Gender (.pt) berbasis YOLO untuk mendeteksi jenis kelamin pada gambar manusia.

Cara Menjalankan di Lokal:
1. Pastikan sudah menginstal dependensi:
   pip install -r requirements.txt
2. Jalankan aplikasi dengan perintah:
   streamlit run app.py

Struktur Folder:
📦 streamlit-multimodel-dashboard
├── model/                -> berisi file model (.h5 dan .pt)
├── sample_images/        -> berisi contoh gambar uji
├── app.py                -> kode utama aplikasi Streamlit
├── requirements.txt      -> daftar library yang dibutuhkan
└── readme.txt            -> deskripsi proyek

Kontributor:
Evan Febrian – Universitas Sriwijaya
