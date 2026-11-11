# 🌴 Klasifikasi Tingkat Kematangan Tandan Buah Segar (TBS) Kelapa Sawit
### ResNet-50 + CBAM (Convolutional Block Attention Module)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Project ini membangun model *deep learning* untuk mengklasifikasikan **tingkat
kematangan buah kelapa sawit** (3 kelas) dari citra. Arsitektur yang digunakan
adalah **ResNet-50** yang dimodifikasi dengan menyisipkan modul **CBAM**
(*Channel Attention* + *Spatial Attention*) pada setiap *residual block*, dengan
bobot awal *transfer learning* dari **ImageNet**.

> Proyek ini merupakan bagian dari penelitian skripsi. Dua skenario pembagian
> data dibandingkan: **80:10:10** dan **90:10**.

---

## 📑 Daftar Isi
- [Fitur Utama](#-fitur-utama)
- [Arsitektur Model](#-arsitektur-model)
- [Struktur Repository](#-struktur-repository)
- [Dataset](#-dataset)
- [Cara Menjalankan](#-cara-menjalankan)
- [Konfigurasi Eksperimen](#-konfigurasi-eksperimen)
- [Hasil](#-hasil)
- [Catatan Penting](#-catatan-penting)
- [Lisensi](#-lisensi)

---

## ✨ Fitur Utama
- **ResNet-50 + CBAM** dibangun manual dengan Keras Functional API.
- CBAM disisipkan **sebelum operasi *Add* (skip connection)** di setiap blok residual.
- **Transfer learning** otomatis: bobot ImageNet disalin ke layer yang namanya
  cocok dengan `keras.applications.ResNet50`.
- Pembagian dataset otomatis dengan `split-folders` (seed tetap = **1337** agar reprodusibel).
- Normalisasi **Z-score ImageNet** dan augmentasi data khusus data latih.
- Callback `ReduceLROnPlateau` dan `EarlyStopping`.
- Evaluasi: *classification report* (precision, recall, F1) dan *confusion matrix*.

## 🏗️ Arsitektur Model

```
Input (224×224×3)
  └─ Conv 7×7 → BN → ReLU → MaxPool
      └─ Stage 2: 3 blok  [64, 64, 256]    ┐
      └─ Stage 3: 4 blok  [128, 128, 512]  │ setiap blok:
      └─ Stage 4: 6 blok  [256, 256, 1024] │ Conv1×1 → Conv3×3 → Conv1×1 → CBAM → Add(shortcut) → ReLU
      └─ Stage 5: 3 blok  [512, 512, 2048] ┘
          └─ Global Average Pooling
              └─ Dense 512 (ReLU) → Dropout 0.5
                  └─ Dense 3 (Softmax)
```

**Modul CBAM** (reduction ratio = 16, kernel spasial = 7×7):
1. *Channel Attention*: GAP + GMP → Shared MLP → penjumlahan → Sigmoid → perkalian.
2. *Spatial Attention*: rata-rata & maksimum sepanjang channel → Concat → Conv 7×7 → Sigmoid → perkalian.

## 📁 Struktur Repository

```
sawit-resnet50-cbam/
├── colab/
│   ├── eksperimen_split_80_10_10.py   # Skenario 80% train / 10% val / 10% test
│   └── eksperimen_split_90_10.py      # Skenario 90% train / 10% val
├── docs/
│   └── PANDUAN_PUBLISH_GITHUB.md
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

> File di folder `colab/` adalah hasil ekspor dari **Google Colab**, sehingga
> berisi perintah khusus Colab (`!pip`, `!unzip`, `drive.mount`). Jalankan
> sebagai notebook di Colab, bukan dengan `python file.py` biasa.

## 🗂️ Dataset

Dataset **tidak disertakan** di repository ini karena ukurannya besar.

- Total: **3.375 citra** dengan **3 kelas** kematangan TBS kelapa sawit
- Kelas: `unripe` (Belum Masak), `ripe` (Masak), `overripe` (Terlalu Masak)
- Sumber dataset: _[isi: dikumpulkan sendiri / link Kaggle / dll.]_

Struktur yang diharapkan di dalam file ZIP (sesuai kode):

```
dataset_sawit.zip
└── dataset final/
    ├── overripe/
    ├── ripe/
    └── unripe/
```

Satu folder = satu kelas. Nama kelas dibaca otomatis dari nama folder
(urutan alfabetis: `overripe`, `ripe`, `unripe`).

## 🚀 Cara Menjalankan

### Opsi A — Google Colab (disarankan, butuh GPU)
1. Upload `dataset_sawit.zip` ke Google Drive, mis. `MyDrive/datasetTBS/dataset_sawit.zip`.
2. Buka Google Colab, buat notebook baru, dan atur runtime ke **GPU** (Runtime → Change runtime type).
3. Salin isi salah satu skrip di folder `colab/` ke sel-sel notebook.
4. Sesuaikan path jika perlu:
   ```python
   drive_zip_path = '/content/drive/MyDrive/datasetTBS/dataset_sawit.zip'
   input_folder = os.path.join(raw_extract_path, 'dataset final')
   ```
5. Jalankan semua sel secara berurutan.

### Opsi B — Lokal
```bash
git clone https://github.com/USERNAME/sawit-resnet50-cbam.git
cd sawit-resnet50-cbam
pip install -r requirements.txt
```
Hapus/ganti baris khusus Colab (`!pip`, `!cp`, `!unzip`, `drive.mount`,
`google.colab.files`) dengan path lokal Anda, lalu jalankan skrip.

## ⚙️ Konfigurasi Eksperimen

| Parameter | Nilai |
|---|---|
| Ukuran citra | 224 × 224 |
| Batch size | 32 |
| Jumlah kelas | 3 |
| Optimizer | Adam (lr = 1e-4, weight decay = 1e-4) |
| Loss | Categorical Crossentropy |
| Epoch maksimum | 100 |
| ReduceLROnPlateau | monitor `val_loss`, factor 0.5, patience 5, min_lr 1e-7 |
| EarlyStopping | monitor `val_loss`, patience 10, `restore_best_weights=True` |
| Seed | 1337 |

### Perbedaan kedua skenario

| Aspek | 80:10:10 | 90:10 |
|---|---|---|
| Pembagian data | 80% train, 10% val, 10% test | 90% train, 10% val |
| Data uji akhir | `test` (terpisah) | `val` (dipakai juga sebagai uji) |
| Translasi acak | ±15% | – |
| Rotasi acak | ±45° | ±30° |
| Flip | horizontal & vertikal | horizontal & vertikal |
| Brightness / Contrast | ±25% | ±20% |
| Saturasi | [0.7, 1.3] | [0.8, 1.2] |
| Hue | ±0.05 | – |

Data validasi dan uji **hanya dinormalisasi** (Z-score ImageNet), tanpa augmentasi.

## 📊 Hasil

Perbandingan performa pada data uji (**339 citra**):

| Kelas | Metrik | Skenario 1 (80:10:10) | Skenario 2 (90:10) |
|---|---|---|---|
| **Belum Masak** (unripe) | Precision | 95,00% | 100,00% |
| | Recall | 99,00% | 100,00% |
| | F1-Score | 97,00% | 100,00% |
| **Masak** (ripe) | Precision | 96,00% | 99,11% |
| | Recall | 95,00% | 98,23% |
| | F1-Score | 95,00% | 98,67% |
| **Terlalu Masak** (overripe) | Precision | 99,00% | 98,25% |
| | Recall | 96,00% | 99,12% |
| | F1-Score | 97,00% | 98,68% |
| **Overall** | Accuracy | 96,00% | **99,12%** |
| | Macro Avg F1-Score | 96,00% | **99,12%** |
| | Total Kesalahan | 12 dari 339 (3,54%) | **3 dari 339 (0,88%)** |

Skenario 90:10 memberikan hasil terbaik dengan akurasi **99,12%**. Lihat juga
[catatan penting](#-catatan-penting) tentang perbedaan cara evaluasi kedua skenario.

Tambahkan gambar grafik training dan confusion matrix (opsional):

```markdown
![Grafik Training](docs/images/training_curve.png)
![Confusion Matrix](docs/images/confusion_matrix.png)
```

## 📝 Catatan Penting
- Pada skenario **90:10**, data `val` dipakai untuk *EarlyStopping*/`ReduceLROnPlateau`
  **sekaligus** sebagai data uji. Karena itu hasilnya bisa sedikit terlalu optimis
  dibanding skenario 80:10:10 yang memiliki data `test` terpisah.
- Data uji berjumlah 339 citra (~10% dari 3.375), sehingga satu kesalahan prediksi setara ±0,29% akurasi.
- Model hasil training (`.h5`) tidak disimpan di repo. Unduh dari tab **Releases**
  atau latih ulang dengan skrip yang tersedia.
- Reproduksibilitas dijaga dengan `seed = 1337`, namun hasil bisa sedikit berbeda
  antar versi TensorFlow/GPU.

## 🛠️ Teknologi
Python · TensorFlow/Keras · NumPy · Pandas · scikit-learn · Matplotlib · Seaborn · OpenCV · split-folders

## 👤 Penulis
**Rizky Afriansyah** — Universitas Muhammadiyah Riau
- GitHub: [@kawankita](https://github.com/Kawankita)
- Email: rizkyafriansyah@gmail.com

## 📄 Lisensi
Didistribusikan di bawah lisensi **MIT**. Lihat file [LICENSE](LICENSE).

## 🙏 Referensi
- He, K. et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR.
- Woo, S. et al. (2018). *CBAM: Convolutional Block Attention Module.* ECCV.
