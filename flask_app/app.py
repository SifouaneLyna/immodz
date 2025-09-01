from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import os
import re
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime

app = Flask(__name__)

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


model, scaler, tfidf, test_mae = None, None, None, 0
try:
    model_path = '../Models/immo_price_prediction_model.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = joblib.load(model_path)
    if not isinstance(model, GradientBoostingRegressor):
        raise TypeError("Loaded model is not a GradientBoostingRegressor")
    print("Model loaded successfully:", type(model))
except Exception as e:
    print(f"Error loading model: {e}")

try:
    scaler_path = '../Models/scaler.pkl'
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    scaler = joblib.load(scaler_path)
    print("Scaler loaded successfully")
except Exception as e:
    print(f"Error loading scaler: {e}")

try:
    tfidf_path = '../Models/tfidf_vectorizer.pkl'
    if not os.path.exists(tfidf_path):
        raise FileNotFoundError(f"TF-IDF vectorizer file not found: {tfidf_path}")
    tfidf = joblib.load(tfidf_path)
    if not isinstance(tfidf, TfidfVectorizer):
        raise TypeError("Loaded TF-IDF is not a TfidfVectorizer")
    print("TF-IDF vectorizer loaded successfully")
except Exception as e:
    print(f"Error loading TF-IDF vectorizer: {e}")

try:
    mae_path = '../Models/test_mae.pkl'
    if not os.path.exists(mae_path):
        raise FileNotFoundError(f"Test MAE file not found: {mae_path}")
    test_mae = joblib.load(mae_path)
    print("Test MAE loaded successfully:", test_mae)
except Exception as e:
    print(f"Error loading test_mae: {e}")


def clean_text(text):
    text = str(text)
    text = re.sub(r'[\u0600-\u06FF]', ' ', text.lower())
    text = re.sub(r'\W+', ' ', text)
    return text.strip()

def format_centimes(price):
    if not isinstance(price, (int, float)) or price <= 0:
        return "0 centimes"
    centimes = price * 100
    if centimes >= 1_000_000_000:
        return f"{centimes / 1_000_000_000:,.3f} milliard de centimes"
    elif centimes >= 1_000_000:
        return f"{centimes / 1_000_000:,.3f} million de centimes"
    else:
        return f"{centimes:,.0f} centimes"

def format_price_with_dots_and_comma(price):
    price_str = f"{price:,.2f}"
    integer_part, decimal_part = price_str.split('.')
    integer_part = integer_part.replace(',', '.')
    return f"{integer_part},{decimal_part}"

@app.route('/', methods=['GET', 'POST'])
def index():
    errors = {}
    result = None
    submitted_info = None
    if request.method == 'POST':
        print("POST request received")
        try:
            surface = request.form.get('surface')
            main_property_type = request.form.get('main_property_type')
            property_type = request.form.get('property_type') if main_property_type == 'Apartment' else main_property_type
            town = request.form.get('town')
            description = request.form.get('description', '').strip()
            print("Form data:", {
                'surface': surface,
                'main_property_type': main_property_type,
                'property_type': property_type,
                'town': town,
                'description': description
            })

            if not surface:
                errors['surface'] = "Surface requise"
            else:
                try:
                    surface = float(surface)
                    if surface < 35:
                        errors['surface'] = "Surface doit être supérieure ou égale à 35 m²"
                    elif surface > 1000:
                        errors['surface'] = "Surface doit être inférieure ou égale à 1000 m²"
                except ValueError:
                    errors['surface'] = "Surface doit être un nombre"

            if not main_property_type:
                errors['main_property_type'] = "Type de propriété requis"
            elif main_property_type not in ['Apartment', 'Villa', 'Terrain']:
                errors['main_property_type'] = f"Type de propriété invalide: {main_property_type}"

            if main_property_type == 'Apartment' and not property_type:
                errors['property_type'] = "Type d'appartement requis"
            elif property_type not in valid_types:
                errors['property_type'] = f"Type de propriété invalide: {property_type}"

            if not town:
                errors['town'] = "Ville requise"
            elif town.lower() not in [t.lower() for t in towns]:
                errors['town'] = f"Ville invalide: {town}"

            if description and (len(description) < 5 or len(description) > 500):
                errors['description'] = "La description doit contenir entre 5 et 500 caractères"

            if model is None or scaler is None or tfidf is None or test_mae == 0:
                errors['general'] = "Erreur: Modèle, scaler, TF-IDF ou MAE non chargé. Vérifiez les fichiers dans C:/Users/Dell/Desktop/immodz/Models/."
                print("Error: Model, scaler, TF-IDF, or test_mae is None")

            if errors:
                print("Validation errors:", errors)
                return render_template('index.html', errors=errors, result=result, submitted_info=submitted_info)

            if description:
                cleaned_description = clean_text(description)
                tfidf_features = tfidf.transform([cleaned_description]).toarray()
            else:
                tfidf_features = np.zeros((1, 200)) 
            tfidf_df = pd.DataFrame(tfidf_features, columns=[f'tfidf_{i}' for i in range(200)])

            # Prepare input data
            input_data = pd.DataFrame({
                'surface': [np.log1p(surface)],
                f'type_{property_type}': [1],
                f'town_{town}': [1]
            })
            input_data = pd.concat([input_data, tfidf_df], axis=1)

            input_data = input_data.reindex(columns=expected_columns, fill_value=0)
            print("Input data columns:", list(input_data.columns))
            print("Expected columns:", expected_columns)
            if list(input_data.columns) != expected_columns:
                raise ValueError("Input data columns do not match expected columns")

            input_scaled = scaler.transform(input_data)
            predicted_price_log = float(model.predict(input_scaled)[0])
            if np.isnan(predicted_price_log) or predicted_price_log <= 0:
                raise ValueError("Invalid prediction from model")

            predicted_price = np.expm1(predicted_price_log)
            min_price_log = predicted_price_log - test_mae
            max_price_log = predicted_price_log + test_mae
            min_price = np.expm1(min_price_log) if min_price_log > 0 else 0
            max_price = np.expm1(max_price_log)

            result = {
                'min_price_raw': float(min_price),
                'predicted_price_raw': float(predicted_price),
                'max_price_raw': float(max_price),
                'min_price': format_price_with_dots_and_comma(min_price),
                'predicted_price': format_price_with_dots_and_comma(predicted_price),
                'max_price': format_price_with_dots_and_comma(max_price),
                'min_price_centimes': format_centimes(min_price),
                'predicted_price_centimes': format_centimes(predicted_price),
                'max_price_centimes': format_centimes(max_price)
            }
            submitted_info = {
                'surface': f"{surface:,.1f}",
                'property_type': f"Appartement ({property_type})" if main_property_type == 'Apartment' else property_type,
                'town': town,
                'description': description if description else ''
            }

            try:
                prediction_data = pd.DataFrame({
                    'surface': [surface],
                    'property_type': [submitted_info['property_type']],
                    'town': [town],
                    'description': [description],
                    'predicted_price': [predicted_price],
                    'min_price': [min_price],
                    'max_price': [max_price],
                    'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
                })
                prediction_data.to_csv('../predictions.csv', mode='a', header=not pd.io.common.file_exists('predictions.csv'), index=False)
            except Exception as e:
                errors['general'] = f"Erreur lors de l'enregistrement: {str(e)}"
                print(f"CSV error: {e}")

        except Exception as e:
            errors['general'] = f"Erreur: {str(e)}"
            print(f"Prediction error: {e}")

    return render_template('index.html', errors=errors, result=result, submitted_info=submitted_info)

if __name__ == '__main__':
    app.run(debug=True)