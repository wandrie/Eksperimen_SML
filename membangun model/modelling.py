import os
import json
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================
# MLFLOW CONFIGURATION
# ============================================

# Set tracking URI ke localhost
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# Set experiment name
mlflow.set_experiment("Sistem Pendeteksi Harga Rumah di California")

# ============================================
# LOAD DATA
# ============================================

def load_processed_data(base_path='california_housing_preprocessing'):
    """Membaca data yang sudah bersih, di-scale, dan di-split oleh notebook"""
    train_path = os.path.join(base_path, "train_processed.csv")
    test_path = os.path.join(base_path, "test_processed.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"❌ Error: File tidak ditemukan di '{base_path}'. "
            f"Pastikan notebook preprocessing sudah dieksekusi."
        )
        
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    # Pisahkan Fitur (X) dan Target (y)
    X_train = df_train.drop(columns=['MedHouseVal'])
    y_train = df_train['MedHouseVal']
    
    X_test = df_test.drop(columns=['MedHouseVal'])
    y_test = df_test['MedHouseVal']
    
    print(f"✅ Data Latih Berhasil Dimuat: {X_train.shape}")
    print(f"✅ Data Uji Berhasil Dimuat  : {X_test.shape}")
    return X_train, X_test, y_train, y_test

# ============================================
# LOG ARTIFACTS
# ============================================

def log_residual_plot(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5, color='purple')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Prediksi Harga')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.savefig('residual_plot.png')
    plt.close()
    mlflow.log_artifact('residual_plot.png')

def log_actual_vs_predicted(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='teal')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Harga Aktual')
    plt.ylabel('Prediksi Harga')
    plt.title('Harga Aktual vs Prediksi Harga')
    plt.savefig('actual_vs_prediksi.png')
    plt.close()
    mlflow.log_artifact('actual_vs_prediksi.png')

def log_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importance')
        plt.bar(range(len(importance)), importance[indices], color='royalblue')
        plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        mlflow.log_artifact('feature_importance.png')
        
        importance_dict = {feature_names[i]: float(importance[i]) for i in range(len(importance))}
        with open('feature_importance.json', 'w') as f:
            json.dump(importance_dict, f, indent=2)
        mlflow.log_artifact('feature_importance.json')

def log_metrics_info(metrics, model_name):
    metrics_info = {
        "model_name": model_name,
        "metrics": metrics,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    with open('metric_info.json', 'w') as f:
        json.dump(metrics_info, f, indent=2)
    mlflow.log_artifact('metric_info.json')

# ============================================
# TRAINING & MLFLOW TRACKING LOGIC
# ============================================
def train_and_log_model(X_train, X_test, y_train, y_test, model, model_name, params=None):
    feature_names = X_train.columns.tolist()
    
    with mlflow.start_run(run_name=model_name):
        if params:
            mlflow.log_params(params)
        
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        metrics = {
            "rmse_train": np.sqrt(mean_squared_error(y_train, y_pred_train)),
            "rmse_test": np.sqrt(mean_squared_error(y_test, y_pred_test)),
            "mae_train": mean_absolute_error(y_train, y_pred_train),
            "mae_test": mean_absolute_error(y_test, y_pred_test),
            "r2_train": r2_score(y_train, y_pred_train),
            "r2_test": r2_score(y_test, y_pred_test)
        }
        
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
        
        # Generate & simpan grafik/file json ke artifak MLflow
        log_residual_plot(y_test, y_pred_test)
        log_actual_vs_predicted(y_test, y_pred_test)
        log_feature_importance(model, feature_names)
        log_metrics_info(metrics, model_name)
        
        print(f" -> {model_name} selesai dievaluasi.")
        return model, metrics

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    print("="*50)
    print("STARTING EXPERIMENT: NO REDUNDANCY MODELLING")
    print("="*50)
    
    # Load data hasil preprocessing
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # Mendefinisikan kandidat model reguler (Tanpa Tuning)
    models = {
        'Linear Regression': (LinearRegression(), {}),
        'Ridge Regression': (Ridge(alpha=1.0), {'alpha': 1.0}),
        'Lasso Regression': (Lasso(alpha=0.01), {'alpha': 0.01}),
        'Random Forest': (RandomForestRegressor(n_estimators=100, random_state=42), 
                         {'n_estimators': 100, 'random_state': 42}),
        'Gradient Boosting': (GradientBoostingRegressor(n_estimators=100, random_state=42),
                            {'n_estimators': 100, 'random_state': 42})
    }
    
    results = {}
    best_model = None
    best_r2 = -np.inf
    best_model_name = ""
    
    # Looping eksperimen ke MLflow
    for model_name, (model, params) in models.items():
        print(f"\n⏳ Menguji model: {model_name}...")
        trained_model, metrics = train_and_log_model(X_train, X_test, y_train, y_test, model, model_name, params)
        results[model_name] = metrics
        
        if metrics['r2_test'] > best_r2:
            best_r2 = metrics['r2_test']
            best_model = trained_model
            best_model_name = model_name
            
    # Simpan berkas model terbaik secara lokal
    output_dir = "saved_models"
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(output_dir, 'best_model.pkl'))
    
    print("\n" + "="*30)
    print("RINGKASAN EVALUASI AKHIR:")
    print("="*20)
    results_df = pd.DataFrame(results).T
    print(results_df[['rmse_test', 'r2_test']].sort_values('r2_test', ascending=False))
    
    print(f"\n🏆 Model Terbaik: {best_model_name}")
    print(f"📈 R² Score Termahal: {best_r2:.4f} ({best_r2*100:.2f}%)")
    print(f"💾 Model sukses diekspor secara lokal ke '{output_dir}/best_model.pkl'")
    print("="*50)