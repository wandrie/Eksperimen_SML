import os
import time
import joblib
import pandas as pd
import numpy as np
from prometheus_client import start_http_server, Counter, Gauge, Histogram

# ============================================
# Menginisialisasi Metrik Prometheus
# ============================================
# Menghitung total request prediksi yang masuk
PREDICTION_COUNT = Counter(
    'model_predictions_total', 
    'Total number of predictions made by the model'
)

# Memantau nilai estimasi harga rumah terakhir
LAST_PREDICTION_VALUE = Gauge(
    'model_last_prediction_value', 
    'The last house price value predicted by the model'
)

# Mengukur latensi / kecepatan pemrosesan model (dalam detik)
PREDICTION_LATENCY = Histogram(
    'model_prediction_latency_seconds', 
    'Time taken to process prediction',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

# ============================================
# Fungsi untuk memuat model yang sudah dilatih
# ============================================
def load_trained_model():
    model_path = os.path.join("..", "Membangun model", "saved_models", "best_model.pkl")
    if not os.path.exists(model_path):
        model_path = os.path.join("Membangun model", "saved_models", "best_model.pkl")
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model tidak ditemukan di: {model_path}")
    return joblib.load(model_path)

# ============================================
# Runtime utama untuk menjalankan Prometheus Exporter
# ============================================
if __name__ == "__main__":
    print("==================================================")
    print("     PROMETHEUS EXPORTER FOR INFERENCE SERVICE    ")
    print("==================================================")
    
    # Load model
    try:
        model = load_trained_model()
        print("✅ Model sukses dimuat untuk kebutuhan exporter.")
    except Exception as e:
        print(e)
        exit(1)

    # Menjalankan HTTP Server internal Prometheus di Port 8000
    PORT = 8000
    start_http_server(PORT)
    print(f"🚀 Exporter aktif! Menyiarkan metrik di http://localhost:{PORT}/metrics")
    print("-" * 50)

    # Loop otomatis untuk mensimulasikan request masuk secara berkala
    request_id = 1
    try:
        while True:
            # Membuat variasi nilai MedInc acak agar grafik pemantauan nanti bersifat fluktuatif (realistis)
            random_medinc = np.random.uniform(2.0, 8.0)
            random_age = np.random.uniform(10.0, 52.0)
            
            data_simulasi = {
                'MedInc': [random_medinc],
                'HouseAge': [random_age],
                'AveRooms': [5.4],
                'AveBedrms': [1.0],
                'Population': [1400.0],
                'AveOccup': [2.8],
                'Latitude': [35.6],
                'Longitude': [-119.5],
                'BedroomsPerRoom': [0.185],
                'IncomeCategory_Encoded': [1]
            }
            df_input = pd.DataFrame(data_simulasi)

            # Memulai Prediksi & Hitung Waktu Proses
            start_time = time.time()
            hasil_prediksi = model.predict(df_input)[0]
            duration = time.time() - start_time

            # Mengirim metrik ke Prometheus
            PREDICTION_COUNT.inc()                  # Untuk menghitung total request prediksi
            LAST_PREDICTION_VALUE.set(hasil_prediksi) # Untuk mengupdate angka harga rumah terbaru
            PREDICTION_LATENCY.observe(duration)     # Untuk Mencatat kecepatan durasi ke histogram

            print(f"[{request_id}] Prediksi Berhasil -> MedInc: {random_medinc:.2f} | Estimasi Harga: ${hasil_prediksi:.4f} | Waktu: {duration:.4f}s")
            
            request_id += 1
            # Request tiruan akan dikirim setiap 3 detik sekali
            time.sleep(3)
            
    except KeyboardInterrupt:
        print("\n🛑 Exporter dihentikan secara manual.")
        print("==================================================")