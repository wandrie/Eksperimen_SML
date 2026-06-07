import os
import joblib
import pandas as pd
import numpy as np

def load_trained_model():
    # Memakai model terbaik
    model_path = os.path.join("..", "Membangun model", "saved_models", "best_model.pkl")
    
    if not os.path.exists(model_path):
        # Fallback jika modelnya berada di root project
        model_path = os.path.join("Membangun model", "saved_models", "best_model.pkl")
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model tidak ditemukan! Pastikan file 'best_model.pkl' ada di {model_path}")
        
    print(f"⏳ Memuat model terbaik dari: {model_path}...")
    model = joblib.load(model_path)
    print("✅ Model Berhasil Dimuat!")
    return model

def main():
    print("==================================================")
    print("      SISTEM INFERENSI PREDIKSI HARGA RUMAH       ")
    print("==================================================")
    
    # 1. Load Model
    try:
        model = load_trained_model()
    except Exception as e:
        print(e)
        return

    # Simulasi Data Baru
    # Kita buat dummy data rumah berdasarkan nilai rata-rata California Housing
    print("\n⏳ Menerima data masukan baru untuk prediksi...")
    
    # Mengganti nilai untuk mencoba prediksi rumah yang berbeda
    data_baru = {
        'MedInc': [4.5],                # Pendapatan rata-rata kelompok (skala puluhan ribu USD)
        'HouseAge': [28.0],             # Umur rumah
        'AveRooms': [5.4],              # Rata-rata jumlah kamar
        'AveBedrms': [1.0],             # Rata-rata jumlah kamar tidur
        'Population': [1400.0],         # Populasi sekitar blok rumah
        'AveOccup': [2.8],              # Rata-rata hunian per rumah
        'Latitude': [35.6],             # Garis lintang lokasi
        'Longitude': [-119.5],          # Garis bujur lokasi
        'BedroomsPerRoom': [0.185],     # Rasio kamar tidur per kamar (hasil Feature Engineering)
        'IncomeCategory_Encoded': [1]   # Hasil encode binning pendapatan (0=Low, 1=Medium, 2=High)
    }
    
    df_input = pd.DataFrame(data_baru)
    
    # Melakukan Prediksi
    print("⏳ Menghitung prediksi menggunakan model Random Forest...")
    hasil_prediksi = model.predict(df_input)
    
    # Menampilkan Hasil
    print("-" * 50)
    print("HASIL PREDIKSI:")
    # Dikali 100,000 jika target aslinya merupakan skala ratusan ribu dolar
    print(f"👉 Estimasi Nilai Rumah: ${hasil_prediksi[0]:.4f} (atau sekitar ${hasil_prediksi[0]*100000:,.2f})")
    print("-" * 50)
    print("✅ Proses Inferensi Selesai dengan Sukses!")
    print("==================================================")

if __name__ == "__main__":
    main()