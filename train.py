import pandas as pd
import numpy as np
import json
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os
# Ensure Models directory exists
os.makedirs('Models', exist_ok=True)
# Verify ads_export.json
try:
    with open('ads_export.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        print(f"Loaded {len(data)} records from ads_export.json")
except Exception as e:
    print(f"Error loading ads_export.json: {e}")
    exit(1)
df = pd.DataFrame(data)
df = df.dropna(subset=['price', 'price_v', 'norm_price'])
df['description'] = df['description'].fillna('No description')
df = df[df['category'].str.startswith('immobilier-vente')]
def extract_property_type(row):
    if row['category'] == 'immobilier-vente-villa':
        return 'Villa'
    elif row['category'] == 'immobilier-vente-terrain':
        return 'Terrain'
    elif row['category'] in ['immobilier-vente-appartement', 'immobilier-vente']:
        match = re.search(r'\b[Ff](0?[1-8])\b', row['title'])
        if match:
            return f'F{match.group(1).lstrip("0")}'
        else:
            return 'Apartment'
    return 'Unknown'
df['property_type'] = df.apply(extract_property_type, axis=1)
def update_property_type(row):
    if row['property_type'] == 'Apartment':
        if isinstance(row['description'], str):
            match = re.search(r'\b[Ff](0?[1-8])\b', row['description'])
            if match:
                return f'F{match.group(1).lstrip("0")}'
        return 'Apartment'
    return row['property_type']
df['property_type'] = df.apply(update_property_type, axis=1)
valid_types = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'Villa', 'Terrain']
# Identify and remove outliers
initial_count = len(df)
outliers = df[
    (df['surface'] <= 20) | (df['surface'] > 1000) |  
    (df['price'] < 1000000) | (df['price'] > 100000000) |  
    (df['town'] == '') | 
    (~df['property_type'].isin(valid_types)) 
]
df = df[
    (df['surface'] > 20) & (df['surface'] <= 1000) &  
    (df['price'] >= 1000000) & (df['price'] <= 100000000) & 
    (df['town'] != '') & 
    (df['property_type'].isin(valid_types)) 
]
print(f"Removed {initial_count - len(df)} outliers, {len(df)} records remain")
key_columns = ['surface', 'price', 'town', 'property_type']
df_cleaned = df[key_columns]
# Encode categorical variables
df_encoded = pd.get_dummies(df_cleaned, columns=['property_type', 'town'], prefix=['type', 'town'])
df_encoded['price'] = np.log1p(df_encoded['price'])
df_encoded['surface'] = np.log1p(df_encoded['surface'])
# Save encoded data
df_encoded.to_csv('encoded_real_estate_data.csv', index=False)
print("Saved encoded_real_estate_data.csv")
# Prepare features and target
scaler = StandardScaler()
X = df_encoded.drop('price', axis=1)
y = df_encoded['price']
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
# Remove outliers from training data
Q1 = y_train.quantile(0.25)
Q3 = y_train.quantile(0.75)
IQR = Q3 - Q1
mask = (y_train >= Q1 - 1.5 * IQR) & (y_train <= Q3 + 1.5 * IQR)
X_train_filtered = X_train_s[mask]
y_train_filtered = y_train[mask]
# Filter test set outliers
mask_test = (y_test >= Q1 - 1.5 * IQR) & (y_test <= Q3 + 1.5 * IQR)
X_test_filtered = X_test_s[mask_test]
y_test_filtered = y_test[mask_test]
# Train model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.2, max_depth=5, random_state=42)
model.fit(X_train_filtered, y_train_filtered)
# Evaluate model
y_pred_train = model.predict(X_train_filtered)
y_pred_test_filtered = model.predict(X_test_filtered)
# Calculate metrics
train_r2 = model.score(X_train_filtered, y_train_filtered)
test_r2 = model.score(X_test_filtered, y_test_filtered)
train_mae = mean_absolute_error(y_train_filtered, y_pred_train)
test_mae = mean_absolute_error(y_test_filtered, y_pred_test_filtered)
train_mse = mean_squared_error(y_train_filtered, y_pred_train)
test_mse = mean_squared_error(y_test_filtered, y_pred_test_filtered)
print("\nGradientBoostingRegressor Results:")
print(f"Train R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")
print(f"Train MAE: {train_mae:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Train MSE: {train_mse:.4f}")
print(f"Test MSE: {test_mse:.4f}")
# Save model, scaler, and test_mae
model_path = 'Models/immo_price_prediction_model.pkl'
joblib.dump(model, model_path)
joblib.dump(scaler, 'Models/scaler.pkl')
joblib.dump(test_mae, 'Models/test_mae.pkl')
# Verify saved files
try:
    test_model = joblib.load(model_path)
    print("Saved model verified successfully:", type(test_model))
    test_scaler = joblib.load('Models/scaler.pkl')
    print("Saved scaler verified successfully:", type(test_scaler))
    test_mae_loaded = joblib.load('Models/test_mae.pkl')
    print("Saved test_mae verified successfully:", test_mae_loaded)
except Exception as e:
    print(f"Error verifying saved files: {e}")