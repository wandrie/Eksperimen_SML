import os
import json
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================
# MLFLOW CONFIGURATION
# ============================================

# Set tracking URI ke localhost
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# Set experiment name (Gunakan eksperimen yang sama agar tercatat berdampingan)
mlflow.set_experiment("Sistem Pendeteksi Harga Rumah di California")

# ============================================
# LOAD DATA
# ============================================

def load_processed_data(base_path='preprocessing'):
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
    
    # Memisahkan Fitur (X) dan Target (y)
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

def log_residual_plot(y_test, y_pred, model_name):
    plt.figure(figsize=(10, 6))
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5, color='darkred')
    plt.axhline(y=0, color='blue', linestyle='--')
    plt.xlabel('Prediksi Harga')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot - {model_name}')
    filename = f'residual_plot_tuning_{model_name.replace(" ", "_")}.png'
    plt.savefig(filename)
    plt.close()
    mlflow.log_artifact(filename)
    os.remove(filename)

def log_actual_vs_predicted(y_test, y_pred, model_name):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='forestgreen')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Harga Aktual')
    plt.ylabel('Prediksi Harga')
    plt.title(f'Harga Aktual vs Prediksi Harga (Tuned {model_name})')
    filename = f'actual_vs_prediksi_tuning_{model_name.replace(" ", "_")}.png'
    plt.savefig(filename)
    plt.close()
    mlflow.log_artifact(filename)
    os.remove(filename)

def log_feature_importance(model, feature_names, model_name):
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title(f'Feature Importance - Tuned {model_name}')
        plt.bar(range(len(importance)), importance[indices], color='gold')
        plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        
        filename_png = f'feature_importance_tuning_{model_name.replace(" ", "_")}.png'
        plt.savefig(filename_png)
        plt.close()
        mlflow.log_artifact(filename_png)
        os.remove(filename_png)
        
        importance_dict = {feature_names[i]: float(importance[i]) for i in range(len(importance))}
        filename_json = f'feature_importance_tuning_{model_name.replace(" ", "_")}.json'
        with open(filename_json, 'w') as f:
            json.dump(importance_dict, f, indent=2)
        mlflow.log_artifact(filename_json)
        os.remove(filename_json)

def log_metrics_info(metrics, model_name):
    metrics_info = {
        "model_name": f"Tuned {model_name}",
        "metrics": metrics,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    filename = f'metric_info_tuning_{model_name.replace(" ", "_")}.json'
    with open(filename, 'w') as f:
        json.dump(metrics_info, f, indent=2)
    mlflow.log_artifact(filename)
    os.remove(filename)

# ============================================
# TUNING & TRACKING LOGIC
# ============================================

def tune_and_log_model(X_train, X_test, y_train, y_test, base_model, model_name, param_grid):
    feature_names = X_train.columns.tolist()
    
    # Inisialisasi GridSearchCV dengan Cross Validation 3-Fold
    grid_search = GridSearchCV(
        estimator=base_model, 
        param_grid=param_grid, 
        cv=3, 
        scoring='r2', 
        n_jobs=-1, 
        verbose=1
    )
    
    print(f"\n🚀 Mencari kombinasi parameter terbaik untuk {model_name}...")
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    print(f"🎯 Parameter Terbaik {model_name}: {best_params}")
    
    # Memulai pencatatan ke MLflow Run khusus Tuning
    run_title = f"Tuned {model_name}"
    with mlflow.start_run(run_name=run_title):
        mlflow.log_params(best_params)
        mlflow.log_param("tuning_method", "GridSearchCV")
        
        # Prediksi hasil
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)
        
        metrics = {
            "rmse_train": np.sqrt(mean_squared_error(y_train, y_pred_train)),
            "rmse_test": np.sqrt(mean_squared_error(y_test, y_pred_test)),
            "mae_train": mean_absolute_error(y_train, y_pred_train),
            "mae_test": mean_absolute_error(y_test, y_pred_test),
            "r2_train": r2_score(y_train, y_pred_train),
            "r2_test": r2_score(y_test, y_pred_test)
        }
        
        # Mengcatat metrik evaluasi ke MLflow
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, "tuned_model")
        
        # Generate & simpan grafik/file json ke artifak MLflow
        log_residual_plot(y_test, y_pred_test, model_name)
        log_actual_vs_predicted(y_test, y_pred_test, model_name)
        log_feature_importance(best_model, feature_names, model_name)
        log_metrics_info(metrics, model_name)
        
        print(f" -> {run_title} sukses dievaluasi dan dicatat ke MLflow.")
        return best_model, metrics

# ============================================
# MAIN 
# ============================================

if __name__ == "__main__":
    print("="*50)
    print("STARTING EXPERIMENT: HYPERPARAMETER TUNING SML")
    print("="*50)
    
    # Load data hasil preprocessing
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # Mendefinisikan kandidat model dan grid parameter untuk tuning
    tuning_candidates = {
        'Random Forest': {
            'model': RandomForestRegressor(random_state=42),
            'grid': {
                'n_estimators': [100, 150],
                'max_depth': [10, 15],  # Pembatasan kedalaman untuk mengurangi overfitting
                'min_samples_split': [5, 10]
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingRegressor(random_state=42),
            'grid': {
                'n_estimators': [100, 150],
                'learning_rate': [0.05, 0.1],
                'max_depth': [4, 6]
            }
        }
    }
    
    tuning_results = {}
    best_tuned_model = None
    best_tuned_r2 = -np.inf
    best_tuned_model_name = ""
    
    # Looping eksekusi pencarian parameter terbaik
    for model_name, config in tuning_candidates.items():
        trained_model, metrics = tune_and_log_model(
            X_train, X_test, y_train, y_test, 
            config['model'], model_name, config['grid']
        )
        tuning_results[model_name] = metrics
        
        if metrics['r2_test'] > best_tuned_r2:
            best_tuned_r2 = metrics['r2_test']
            best_tuned_model = trained_model
            best_tuned_model_name = f"Tuned {model_name}"
            
    # Simpan berkas model terbaik hasil tuning ke folder lokal tugas
    output_dir = "saved_models"
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(best_tuned_model, os.path.join(output_dir, 'best_model.pkl'))
    
    print("\n" + "="*30)
    print("RINGKASAN EVALUASI AKHIR (HYPERPARAMETER TUNING):")
    print("="*20)
    results_df = pd.DataFrame(tuning_results).T
    print(results_df[['rmse_test', 'r2_test']].sort_values('r2_test', ascending=False))
    
    print(f"\n🏆 Model Terbaik Hasil Tuning: {best_tuned_model_name}")
    print(f"📈 R² Score Tertinggi: {best_tuned_r2:.4f} ({best_tuned_r2*100:.2f}%)")
    print(f"💾 Model sukses memperbarui '{output_dir}/best_model.pkl' dengan versi parameter optimal!")
    print("="*50)