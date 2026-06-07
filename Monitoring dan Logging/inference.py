import joblib
import numpy as np
import pandas as pd
import time
import os
import sys
from datetime import datetime

# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Flask untuk API 
try:
    from flask import Flask, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("⚠️ Flask tidak tersedia. Install dengan: pip install flask")

# ============================================
# KONFIGURASI PATH
# ============================================

# PATH CONFIGURATION
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
model_dir = os.path.join(project_dir, 'Membangun model')

# Path ke model dan scaler
model_path = os.path.join(model_dir, 'best_model.pkl')
scaler_path = os.path.join(model_dir, 'scaler.pkl')

# ============================================
# PROMETHEUS METRICS
# ============================================

# Counter untuk jumlah prediksi
prediction_total = Counter(
    'inference_prediction_total',
    'Total number of predictions made'
    )

# Counter untuk error
error_total = Counter(
    'inference_error_total',
    'Total number of prediction errors'
    )

# Histogram untuk latency
prediction_latency = Histogram(
    'inference_prediction_latency_seconds',
    'Time taken for prediction in seconds'
    )

# Gauge untuk nilai prediksi terakhir
last_prediction = Gauge(
    'inference_last_prediction_value',
    'Last prediction value'
    )

# Counter berdasarkan kategori harga
price_category = Counter(
    'inference_price_category_total',
    'Predictions by price category',
    ['category']
    )

# Gauge untuk model info
model_info = Gauge(
    'inference_model_info',
    'Model information',
    ['model_type', 'version']
    )

# ============================================
# LOAD MODEL
# ============================================

def load_model():
    """Load model dan scaler yang sudah dilatih"""
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print("Model dan scaler berhasil dimuat")
        print(f"   Model path: {model_path}")
        print(f"   Scaler path: {scaler_path}")
        
        # Set model info metric
        model_info.labels(model_type='RandomForest', version='1.0').set(1)
        
        return model, scaler
    except FileNotFoundError as e:
        print(f"❌ Error loading model: {e}")
        print("   file tidak ditemukan. Pastikan model sudah dilatih dan file 'best_model.pkl' serta 'scaler.pkl' ada di folder 'Membangun model'.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

# ============================================
# PREDICTION FUNCTION
# ============================================

# Nama fitur sesuai dengan yang digunakan saat training
FEATURE_NAMES = [
    'MedInc',
    'HouseAge',
    'AveRooms',
    'AveBedrms', 
    'Population',
    'AveOccup',
    'Latitude',
    'Longitude',
    'RoomsPerHousehold',
    'BedroomsPerRoom',
    'PopulationPerHousehold'
]

def predict(features, model, scaler):
    """
    Melakukan prediksi harga rumah
    
    Args:
        features: list atau array dengan 11 nilai fitur
        model: loaded model
        scaler: loaded scaler
    
    Returns:
        float: predicted house price (in $100,000 units)
    """
    start_time = time.time()
    
    try:
        # Konversi ke numpy array
        features_array = np.array(features).reshape(1, -1)
        
        # Standarisasi fitur
        features_scaled = scaler.transform(features_array)
        
        # Prediksi
        prediction = model.predict(features_scaled)[0]
        
        # Update metrics
        prediction_total.inc()
        last_prediction.set(prediction)
        prediction_latency.observe(time.time() - start_time)
        
        # Update category counter
        if prediction < 1:
            price_category.labels(category='Very Low').inc()
        elif prediction < 2:
            price_category.labels(category='Low').inc()
        elif prediction < 3:
            price_category.labels(category='Medium').inc()
        elif prediction < 4:
            price_category.labels(category='High').inc()
        elif prediction < 5:
            price_category.labels(category='Very High').inc()
        else:
            price_category.labels(category='Extreme').inc()
        
        return prediction
        
    except Exception as e:
        error_total.inc()
        print(f"❌ Prediction error: {e}")
        return None

def predict_with_interpretation(
        features,
        model,
        scaler
        ):
    
    """Memprediksi harga rumah dengan interpretasi kategori dan deskripsi"""
    price = predict(
        features,
        model,
        scaler
        )
    
    if price is None:
        return None, None
    
    if price < 1:
        category = "Very Low (< $100,000)"
        description = "Harga sangat murah, kemungkinan daerah kurang strategis"
    elif price < 2:
        category = "Low ($100,000 - $200,000)"
        description = "Harga terjangkau, cocok untuk pembeli pertama"
    elif price < 3:
        category = "Medium ($200,000 - $300,000)"
        description = "Harga menengah, area kelas menengah"
    elif price < 4:
        category = "High ($300,000 - $400,000)"
        description = "Harga tinggi, lokasi premium"
    elif price < 5:
        category = "Very High ($400,000 - $500,000)"
        description = "Harga sangat tinggi, area eksklusif"
    else:
        category = "Extreme (> $500,000)"
        description = "Harga ekstrem, kemungkinan area mewah atau pusat kota"
    
    return price, category, description

# ============================================
# BATCH PREDICTION
# ============================================

def batch_predict(
        features_list,
        model,
        scaler
        ):
    """Memprediksi harga untuk banyak data sekaligus"""
    predictions = []
    for features in features_list:
        pred = predict(features, model, scaler)
        predictions.append(pred)
    return predictions

# ============================================
# PREDICT FROM CSV
# ============================================

def predict_from_csv(csv_path,
                     model,
                     scaler
                     ):
    """Prediksi dari file CSV"""
    df = pd.read_csv(csv_path)
    
    # Pastikan kolom yang diperlukan ada
    missing_cols = [col for col in FEATURE_NAMES if col not in df.columns]
    if missing_cols:
        print(f"Kolom yang hilang: {missing_cols}")
        return None
    
    X = df[FEATURE_NAMES].values
    predictions = []
    
    for features in X:
        pred = predict(features.tolist(), model, scaler)
        predictions.append(pred)
    
    df['Prediction'] = predictions
    return df

# ============================================
# FLASK API (OPSIONAL)
# ============================================

def create_app(model, scaler):
    """Buat Flask app untuk REST API"""
    if not FLASK_AVAILABLE:
        return None
    
    app = Flask(__name__)
    
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({
            'status': 'healthy',
            'model_loaded': model is not None,
            'timestamp': datetime.now().isoformat()
        })
    
    @app.route('/predict', methods=['POST'])
    def predict_api():
        try:
            data = request.get_json()
            features = data.get('features')
            
            if not features:
                return jsonify({'error': 'No features provided'}), 400
            
            if len(features) != 11:
                return jsonify({'error': f'Expected 11 features, got {len(features)}'}), 400
            
            price = predict(features, model, scaler)
            
            if price is None:
                return jsonify({'error': 'Prediction failed'}), 500
            
            return jsonify({
                'prediction': price,
                'prediction_usd': price * 100000,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/predict/batch', methods=['POST'])
    def predict_batch_api():
        try:
            data = request.get_json()
            features_list = data.get('features_list')
            
            if not features_list:
                return jsonify({'error': 'No features provided'}), 400
            
            predictions = []
            for features in features_list:
                price = predict(features, model, scaler)
                predictions.append({
                    'features': features,
                    'prediction': price,
                    'prediction_usd': price * 100000 if price else None
                })
            
            return jsonify({
                'predictions': predictions,
                'total': len(predictions),
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return app

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("="*30)
    print("HOUSE PRICE PREDICTION - INFERENCE SERVICE")
    print("="*30)
    
    # Load model
    model, scaler = load_model()
    
    # Menjalankan Prometheus metrics di port 8001
    start_http_server(8001)
    print("✅ Prometheus metrics available at http://localhost:8001/metrics")
    
    # Contoh prediksi tunggal
    print("\n" + "-"*20)
    print("SAMPLE PREDICTION")
    print("-"*20)
    
    sample_features = [3.0,
                       30.0,
                       5.0,
                       1.0,
                       1000.0,
                       2.5,
                       34.0,
                       -118.0,
                       2.0,
                       0.2,
                       400.0
                       ]
    
    price, category, description = predict_with_interpretation(sample_features, model, scaler)
    
    if price:
        print(f"Input features: {sample_features}")
        print(f"\nPrediksi Harga Rumah:")
        print(f"   Harga: ${price * 100000:,.2f}")
        print(f"   Kategori: {category}")
        print(f"   Deskripsi: {description}")
    
    # Menjalankan Flask API jika tersedia
    if FLASK_AVAILABLE:
        print("\n" + "-"*20)
        print("MENJALANKAN SERVER FLASK API")
        print("-"*20)
        app = create_app(model, scaler)
        print("Flask API berjalan di http://localhost:5000")
        print("   Endpoints:")
        print("     GET  /health          - Health check")
        print("     POST /predict         - Single prediction")
        print("     POST /predict/batch   - Batch prediction")
        
        # Run Flask app
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("\nFlask not installed. Install with: pip install flask")
        print("Keeping script running for metrics...")
        
        try:
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            print("\n Inference service stopped by user.")