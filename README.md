# 🏡 Sistem Pendeteksi Harga Rumah di California (End-to-End MLOps Pipeline)

Proyek ini merupakan implementasi pipa MLOps (_Machine Learning Operations_) lengkap dari hulu ke hilir, mulai dari tahap _preprocessing_ data, eksperimen otomatis & _hyperparameter tuning_ menggunakan MLflow, otomatisasi pengujian dengan GitHub Actions (CI), hingga tahap produksi berupa _serving_ model kustom dan _monitoring_ real-time berbasis Prometheus & Grafana.

---

## Pengembang

- **Nama:** Hervan Wandri
- **Peran:** MLOps / Data Engineer
- **Lingkungan Sistem:** macOS (MacBook Pro 2015) & Python 3.11

---

## 📁 Struktur Direktori Proyek

```text
SMSML_Wandrie/
├── README.md                           <- Panduan utama proyek (File ini)
├── Eksperimen_SML_Wandrie.txt          <- Laporan metrik & kesimpulan model terbaik
├── Workflow-CI.txt                     <- Dokumentasi alur kerja otomatisasi CI
├── Screenhoot_artifak.jpg
├── California_raw/
│   └── california_housing.csv          <- Berisi file dataset yang digunakan
├── Membangun_model/
│   ├── modelling.py                    <- Base training (5 algoritma dasar)
│   ├── modelling_tuning.py             <- Hyperparameter Tuning (GridSearchCV)
│   ├── requirements.txt                <- Daftar dependensi pustaka Python
│   └── screenshoot_dashboard.jpg       <- Bukti antarmuka MLflow Tracking
├── Monitoring dan Logging/
│    ├── 1.bukti_serving                 <- Log/bukti endpoint exporter aktif
│    ├── 2.prometheus.yml                <- Konfigurasi target scraping Prometheus
│    ├── 3.prometheus_exporter.py        <- Script kustom penyiaran metrik (psutil)
│    ├── 4.bukti monitoring Prometheus/  <- Screenshot metrik pada Prometheus UI
│    ├── 5.bukti monitoring Grafana/     <- Screenshot dashboard visualisasi RAM/CPU
│    ├── 6.bukti alerting Grafana/       <- Screenshot pemicuan alarm (Alerting Firing)
│    └── 7.Inference.py                  <- Script pengujian prediksi data tunggal
└── Membangun_model/
   ├── california_housing_preprocessing/ <- Dataset hasil pembersihan & scaling
   ├── Eksperimen_Hervan-Wandri.ipynb    <- Berisi file eksperimen .ipynb
   ├── test_processed.csv                <- File test yang telah di ekspor
   └── train_processed.csv               <- File train_processed yang telah di ekspor
```
