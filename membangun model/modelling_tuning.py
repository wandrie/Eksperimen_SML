import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ============================================
# LOAD DATA DAN SCALER
# ============================================
def load_data_and_scaler():
    """Memuat data dan scaler dari modelling.py"""
    from sklearn.model_selection import train_test_split
    
    df = pd.read_csv('california_housing_preprocessing/california_housing_processed.csv')
    
    # Hapus kolom binning
    drop_cols = ['PriceCategory', 'IncomeCategory']
    X = df.drop(columns=['MedHouseVal'] + drop_cols, errors='ignore')
    y = df['MedHouseVal']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Load scaler dan transform
    scaler = joblib.load('scaler.pkl')
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# ============================================
# TUNING RANDOM FOREST
# ============================================
def tune_random_forest(X_train, y_train):
    """Hyperparameter tuning untuk Random Forest"""
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING - RANDOM FOREST")
    print("="*70)
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, 
        scoring='neg_mean_squared_error',
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\n✅ Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {-grid_search.best_score_:.4f} (RMSE)")
    
    return grid_search.best_estimator_

# ============================================
# TUNING GRADIENT BOOSTING
# ============================================
def tune_gradient_boosting(X_train, y_train):
    """Hyperparameter tuning untuk Gradient Boosting"""
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING - GRADIENT BOOSTING")
    print("="*70)
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10]
    }
    
    gb = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(
        gb, param_grid, cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\n✅ Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {-grid_search.best_score_:.4f} (RMSE)")
    
    return grid_search.best_estimator_

# ============================================
# EVALUATE TUNED MODEL
# ============================================
def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluasi model setelah tuning"""
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📊 {model_name} - Final Evaluation:")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   R² Score: {r2:.4f}")
    
    return rmse, r2

# ============================================
# SAVE TUNED MODEL
# ============================================
def save_tuned_model(model, filename='best_model_tuned.pkl'):
    joblib.dump(model, filename)
    print(f"✅ Tuned model saved as '{filename}'")

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    print("="*70)
    print("HOUSE PRICE PREDICTION - HYPERPARAMETER TUNING")
    print("="*70)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data_and_scaler()

    # Tuning model dasar (Random Forest dan Gradient Boosting) yang sudah dipilih dari modelling.py
    # Tuning Random Forest
    best_rf = tune_random_forest(X_train, y_train)
    evaluate_model(best_rf, X_test, y_test, "Random Forest (Tuned)")
    
    # Tuning Gradient Boosting
    best_gb = tune_gradient_boosting(X_train, y_train)
    evaluate_model(best_gb, X_test, y_test, "Gradient Boosting (Tuned)")
    
    # Pilih yang terbaik
    rf_rmse, rf_r2 = evaluate_model(best_rf, X_test, y_test, "Random Forest (Tuned)")
    gb_rmse, gb_r2 = evaluate_model(best_gb, X_test, y_test, "Gradient Boosting (Tuned)")
    
    if rf_rmse < gb_rmse:
        best_model = best_rf
        print("\n🏆 Best Tuned Model: Random Forest")
    else:
        best_model = best_gb
        print("\n🏆 Best Tuned Model: Gradient Boosting")
    
    # Save tuned model
    save_tuned_model(best_model, 'best_model_tuned.pkl')
    
    print("\n" + "="*70)
    print("TUNING COMPLETED!")
    print("="*70)