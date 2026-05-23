# 🪙 Prediksi Harga Logam Mulia — Metode Adaline

Aplikasi web prediksi harga **Logam Mulia (LM)** per gram menggunakan metode **Adaline (Adaptive Linear Neuron)**. Dibangun dengan HTML, CSS, dan JavaScript murni — semua proses berjalan di sisi klien (browser).

## 📋 Deskripsi

Aplikasi ini menganalisis data historis harga emas Antam per gram dari tahun **2010 hingga 2026** (5000+ titik data) dan menggunakan jaringan saraf tiruan Adaline untuk memprediksi harga hari berikutnya.

### Fitur Utama

- 📈 **Visualisasi Data Historis** — Grafik interaktif harga LM dari waktu ke waktu
- ⚙️ **Konfigurasi Training** — Atur learning rate, epoch, window size, dan rasio data
- 📊 **Hasil Training Lengkap** — Grafik konvergensi MSE, tabel bobot & bias, metrik evaluasi
- 🔮 **Prediksi Harga** — Prediksi otomatis dan input kustom
- 📱 **Responsif** — Tampilan optimal di desktop dan mobile
- 🌙 **Dark Mode Premium** — Desain modern dengan tema emas

## 🧠 Tentang Metode Adaline

**Adaline (Adaptive Linear Neuron)** dikembangkan oleh Bernard Widrow dan Ted Hoff pada tahun 1960. Merupakan jaringan saraf tiruan satu lapisan yang menggunakan:

- **Fungsi aktivasi linear** (bukan step function seperti Perceptron)
- **Widrow-Hoff Learning Rule** (Least Mean Squares / LMS)
- **Gradient descent** untuk meminimalkan error

### Rumus Pembaruan Bobot

```
w_i(t+1) = w_i(t) + α × (target − net) × x_i
bias(t+1) = bias(t) + α × (target − net)
```

Dimana:
- `α` — Learning rate (kecepatan belajar)
- `target` — Nilai output yang diharapkan
- `net` — Output jaringan: Σ(w_i × x_i) + bias
- `x_i` — Input ke-i

### Pendekatan Sliding Window

Aplikasi ini menggunakan teknik **sliding window** dimana harga `N` hari sebelumnya digunakan sebagai input untuk memprediksi harga hari berikutnya:

```
Input:  [harga(t-N), harga(t-N+1), ..., harga(t-1)]
Output: harga(t)
```

### Normalisasi Data

Data dinormalisasi menggunakan **Min-Max Scaling** ke rentang [0, 1]:

```
x_norm = (x - x_min) / (x_max - x_min)
```

## 🚀 Cara Menjalankan

### Metode 1: Langsung buka di browser

```bash
# Cukup buka file index.html di browser
open index.html
# atau
xdg-open index.html
```

> ⚠️ Beberapa browser memblokir fetch() pada protokol `file://`. Gunakan metode 2 jika terjadi error.

### Metode 2: Menggunakan Live Server

```bash
# Dengan Python
python3 -m http.server 8000

# Dengan Node.js (npx)
npx serve .

# Dengan PHP
php -S localhost:8000
```

Kemudian buka `http://localhost:8000` di browser.

## ⚙️ Parameter Training

| Parameter | Default | Rentang | Deskripsi |
|-----------|---------|---------|-----------|
| **Learning Rate (α)** | 0.01 | 0.0001 – 1 | Kecepatan belajar. Nilai lebih kecil = lebih stabil tapi lebih lambat |
| **Epoch** | 100 | 1 – 10000 | Jumlah iterasi pelatihan |
| **Window Size** | 5 | 2 – 30 | Jumlah hari sebelumnya yang dijadikan input |
| **Rasio Training** | 80% | 50% – 95% | Persentase data untuk training, sisanya untuk testing |

### Tips Pengaturan Parameter

- **Learning rate terlalu besar** → Model tidak konvergen (MSE naik-turun)
- **Learning rate terlalu kecil** → Training sangat lambat
- **Epoch terlalu sedikit** → Model belum konvergen (underfitting)
- **Window size terlalu besar** → Overfitting, model terlalu kompleks
- **Rekomendasi**: Mulai dengan default, lalu sesuaikan learning rate dan epoch

## 📊 Metrik Evaluasi

- **MSE (Mean Squared Error)** — Rata-rata kuadrat error pada data testing
- **MAPE (Mean Absolute Percentage Error)** — Rata-rata persentase error absolut
- **Akurasi** — Dihitung sebagai `100% - MAPE`

## 📁 Struktur File

```
lm-price-prediction-adaline/
├── index.html      # Halaman utama aplikasi
├── style.css       # Stylesheet (dark mode, responsive)
├── app.js          # Logika Adaline, chart, dan interaksi
├── data.csv        # Data historis harga LM (5000+ baris)
└── README.md       # Dokumentasi (file ini)
```

## 🛠️ Teknologi

- **HTML5** — Struktur semantik
- **CSS3** — Glassmorphism, dark mode, animasi, responsive
- **JavaScript (ES6+)** — Implementasi Adaline, data processing
- **Chart.js 4** — Visualisasi grafik interaktif
- **Google Fonts** — Inter & JetBrains Mono

## 📚 Referensi

- Widrow, B., & Hoff, M. E. (1960). *Adaptive Switching Circuits*. IRE WESCON Convention Record.
- Fausett, L. (1994). *Fundamentals of Neural Networks*. Prentice-Hall.
- Haykin, S. (2009). *Neural Networks and Learning Machines* (3rd ed.). Pearson.

## 📄 Lisensi

Proyek ini dibuat untuk keperluan akademik — Tugas Mata Kuliah Jaringan Saraf Tiruan.
