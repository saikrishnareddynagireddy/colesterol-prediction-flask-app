import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import joblib
import os

# ================== LOAD AND CLEAN DATA ==================
df = pd.read_csv("cardio_train.csv")

# Basic cleaning
df['age'] = (df['age'] / 365).round(1)
df.drop(['id'], axis=1, inplace=True, errors='ignore')
df = df[df['cholesterol'] != 1]
df['cholesterol'] = df['cholesterol'].map({2: 200, 3: 240})

# Features and target
X = df[['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'gluc', 'smoke', 'alco', 'active']]
y = df['cholesterol']

# ================== SPLIT DATA ==================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================== SCALE FEATURES ==================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================== DEFINE MULTIPLE MODELS ==================
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.01),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

# ================== TRAIN AND EVALUATE ALL MODELS ==================
results = []

print("🔹 Training and evaluating models...\n")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results.append((name, r2, rmse))
    print(f"{name}: R² = {r2:.3f}, RMSE = {rmse:.3f}")

# ================== SELECT BEST MODEL ==================
results_df = pd.DataFrame(results, columns=["Model", "R2_Score", "RMSE"])
best_model_name = results_df.sort_values(by="R2_Score", ascending=False).iloc[0]["Model"]
best_model = models[best_model_name]

print("\n✅ Best Model Selected:", best_model_name)

# ================== SAVE BEST MODEL AND SCALER ==================
os.makedirs("model", exist_ok=True)
joblib.dump(best_model, f"model/{best_model_name.replace(' ', '_').lower()}_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print(f"💾 Saved Best Model as: model/{best_model_name.replace(' ', '_').lower()}_model.pkl")
print("💾 Scaler saved successfully!")
