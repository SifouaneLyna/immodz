import pandas as pd
import numpy as np
import json
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

nltk.download('stopwords')

os.makedirs('Models', exist_ok=True)

# Define valid types and towns
valid_types = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'Terrain', 'Villa']
towns = [
    'Ain benian', 'Ain naadja', 'Ain taya', 'Alger centre', 'Bab el oued', 'Bab ezzouar',
    'Baba hassen', 'Bachdjerrah', 'Baraki', 'Belouizdad', 'Ben aknoun', 'Beni messous',
    'Bir mourad rais', 'Birkhadem', 'Birtouta', 'Bologhine', 'Bordj el bahri',
    'Bordj el kiffan', 'Bourouba', 'Bouzareah', 'Casbah', 'Cheraga', 'Chevalley',
    'Dar el beida', 'Dely brahim', 'Douera', 'Draria', 'El achour', 'El biar',
    'El harrach', 'El madania', 'El magharia', 'El marsa', 'El mouradia',
    'Gue de constantine', 'Hammamet', 'Hraoua', 'Hussein dey', 'Hydra', 'Khraissia',
    'Kouba', 'Les eucalyptus', 'Mahelma', 'Mohammadia', 'Oued koriche', 'Oued smar',
    'Ouled chebel', 'Ouled fayet', 'Rahmania', 'Rais hamidou', 'Reghaia', 'Rouiba',
    'Said hamdine', 'Saoula', 'Sidi mhamed', 'Sidi moussa', 'Souidania', 'Staoueli',
    'Tessala el merdja', 'Zeralda'
]
expected_columns = ['surface'] + [f'type_{ptype}' for ptype in valid_types] + \
                  [f'town_{town}' for town in towns] + [f'tfidf_{i}' for i in range(200)]

try:
    with open('data/ads_export.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        print(f"Loaded {len(data)} records from data/ads_export.json")
except Exception as e:
    print(f"Error loading data/ads_export.json: {e}")
    exit(1)

df = pd.DataFrame(data)
df = df.dropna(subset=['price', 'price_v', 'norm_price'])
df['description'] = df['description'].fillna('')
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
    if row['property_type'] == 'Apartment' and isinstance(row['description'], str):
        match = re.search(r'\b[Ff](0?[1-8])\b', row['description'])
        if match:
            return f'F{match.group(1).lstrip("0")}'
        return 'Apartment'
    return row['property_type']

df['property_type'] = df.apply(update_property_type, axis=1)

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
key_columns = ['surface', 'price', 'town', 'property_type', 'description']
df_cleaned = df[key_columns].copy()

def clean_text(text):
    text = str(text)
    text = re.sub(r'[\u0600-\u06FF]', ' ', text.lower()) 
    text = re.sub(r'\W+', ' ', text)
    return text.strip()

df_cleaned['cleaned_description'] = df_cleaned['description'].apply(clean_text)

french_stop_words = stopwords.words('french')
custom_stop_words = ['merci', 'contactez', 'agence', 'informations', 'vente', 'vendre', 'vend']
all_stop_words = french_stop_words + custom_stop_words
tfidf = TfidfVectorizer(max_features=200, min_df=5, max_df=0.8, stop_words=all_stop_words)
tfidf_matrix = tfidf.fit_transform(df_cleaned['cleaned_description'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f'tfidf_{i}' for i in range(200)], index=df_cleaned.index)

df_encoded = pd.concat([df_cleaned[['surface']], pd.get_dummies(df_cleaned['property_type'], prefix='type'), pd.get_dummies(df_cleaned['town'], prefix='town'), tfidf_df], axis=1)
df_encoded['price'] = np.log1p(df_cleaned['price'])
df_encoded['surface'] = np.log1p(df_cleaned['surface'])

df_encoded = df_encoded.reindex(columns=expected_columns + ['price'], fill_value=0)

df_encoded.to_csv('encoded_real_estate_data.csv', index=False)
print("Saved encoded_real_estate_data.csv")

scaler = StandardScaler()
X = df_encoded.drop('price', axis=1)
y = df_encoded['price']

model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.2, max_depth=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

Q1 = y_train.quantile(0.25)
Q3 = y_train.quantile(0.75)
IQR = Q3 - Q1
mask = (y_train >= Q1 - 1.5 * IQR) & (y_train <= Q3 + 1.5 * IQR)
X_train_filtered = X_train_s[mask]
y_train_filtered = y_train[mask]

mask_test = (y_test >= Q1 - 1.5 * IQR) & (y_test <= Q3 + 1.5 * IQR)
X_test_filtered = X_test_s[mask_test]
y_test_filtered = y_test[mask_test]

model.fit(X_train_filtered, y_train_filtered)

importances = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
print("\nTop 20 Feature Importances:")
print(importances.head(20))
tfidf_importance = importances[importances['feature'].str.startswith('tfidf_')]['importance'].sum()
print(f"Total TF-IDF Importance: {tfidf_importance:.4f}")

y_pred_train = model.predict(X_train_filtered)
y_pred_test_filtered = model.predict(X_test_filtered)

train_r2 = model.score(X_train_filtered, y_train_filtered)
test_r2 = model.score(X_test_filtered, y_test_filtered)
train_mae = mean_absolute_error(y_train_filtered, y_pred_train)
test_mae = mean_absolute_error(y_test_filtered, y_pred_test_filtered)
train_mse = mean_squared_error(y_train_filtered, y_pred_train)
test_mse = mean_squared_error(y_test_filtered, y_pred_test_filtered)

print("\nGradientBoostingRegressor Results (Single Split):")
print(f"Train R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")
print(f"Train MAE: {train_mae:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Train MSE: {train_mse:.4f}")
print(f"Test MSE: {test_mse:.4f}")

model_path = 'Models/immo_price_prediction_model.pkl'
joblib.dump(model, model_path)
joblib.dump(scaler, 'Models/scaler.pkl')
joblib.dump(tfidf, 'Models/tfidf_vectorizer.pkl')
joblib.dump(test_mae, 'Models/test_mae.pkl')
joblib.dump(expected_columns, 'Models/expected_columns.pkl')

try:
    test_model = joblib.load(model_path)
    print("Saved model verified successfully:", type(test_model))
    test_scaler = joblib.load('Models/scaler.pkl')
    print("Saved scaler verified successfully:", type(test_scaler))
    test_tfidf = joblib.load('Models/tfidf_vectorizer.pkl')
    print("Saved TF-IDF verified successfully:", type(test_tfidf))
    test_mae_loaded = joblib.load('Models/test_mae.pkl')
    print("Saved test_mae verified successfully:", test_mae_loaded)
    test_columns = joblib.load('Models/expected_columns.pkl')
    print("Saved expected columns verified successfully:", len(test_columns), "columns")
except Exception as e:
    print(f"Error verifying saved files: {e}")