import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================
# LOAD DATA
# ============================================
def load_data(filepath='california_housing_preprocessing/california_housing_processed.csv'):
    df = pd.read_csv(filepath)
    print(f"Data loaded: {df.shape}")
    return df

# ============================================
# PREPARE FEATURES
# ============================================
def prepare_features(df, target_col='MedHouseVal', drop_cols=None):
    if drop_cols is None:
        drop_cols = ['PriceCategory', 'IncomeCategory']
    
    X = df.drop(columns=[target_col] + drop_cols, errors='ignore')
    y = df[target_col]
    
    print(f"Fitur (X): {X.shape}")
    print(f"Fitur yang digunakan: {list(X.columns)}")
    return X, y

# ============================================
# SPLIT DATA
# ============================================
def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    return X_train, X_test, y_train, y_test

# ============================================
# STANDARISASI
# ============================================
def standardize_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, 'scaler.pkl')
    print("Scaler saved as 'scaler.pkl'")
    return X_train_scaled, X_test_scaled, scaler

# ============================================
# TRAIN & EVALUATE MODELS
# ============================================
def train_and_evaluate(X_train, y_train, X_test, y_test):
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.01),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1)
    }
    
    results = []
    
    print("\n" + "="*40)
    print("KOMPARASI MODEL")
    print("="*40)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        })
        
        print(f"{name:20} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")
    
    results_df = pd.DataFrame(results).sort_values('RMSE')
    return results_df, models

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    print("="*40)
    print("PREDIKSI HARGA RUMAH")
    print("="*40)
    
    # Load data
    df = load_data('california_housing_preprocessing/california_housing_processed.csv')
    
    # Prepare features
    X, y = prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Standarisasi
    X_train_scaled, X_test_scaled, scaler = standardize_features(X_train, X_test)
    
    # Train and evaluate
    results_df, models = train_and_evaluate(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Best model
    best_model_name = results_df.iloc[0]['Model']
    best_model = models[best_model_name]
    
    # Save best model
    joblib.dump(best_model, 'best_model.pkl')
    print(f"\n Best Model: {best_model_name}")
    print("Model tersimpan sebagai 'best_model.pkl'")
    
    print("\n" + "="*25)
    print("MODELLING COMPLETED!")
    print("="*25)