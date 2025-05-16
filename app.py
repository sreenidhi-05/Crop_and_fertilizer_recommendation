from flask import Flask, render_template, request
import pickle
import numpy as np
from datetime import datetime
import csv

app = Flask(__name__)

with open('crop_model.sav', 'rb') as f:
    crop_model = pickle.load(f)
with open('crop_scaler.sav', 'rb') as f:
    crop_scaler = pickle.load(f)
with open('fertilizer_model.sav', 'rb') as f:
    fertilizer_model = pickle.load(f)
with open('fertilizer_scaler.sav', 'rb') as f:
    fertilizer_scaler = pickle.load(f)

crop_dict = {
    1: 'Rice', 2: 'Maize', 3: 'Jute', 4: 'Cotton', 5: 'Coconut', 6: 'Papaya',
    7: 'Orange', 8: 'Apple', 9: 'Muskmelon', 10: 'Watermelon', 11: 'Grapes',
    12: 'Mango', 13: 'Banana', 14: 'Pomegranate', 15: 'Lentil', 16: 'Blackgram',
    17: 'Mungbean', 18: 'Mothbeans', 19: 'Pigeonpeas', 20: 'Kidneybeans',
    21: 'Chickpea', 22: 'Coffee'
}

fertilizer_dict = {
    0: 'Urea', 1: 'DAP', 2: '14-35-14', 3: '28-28', 4: '17-17-17', 5: '20-20',
    6: '10-26-26'
}

def log_prediction(crop_prediction=None, fertilizer_prediction=None):
    with open('predictions_log.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            crop_prediction or "",
            fertilizer_prediction or ""
        ])

@app.route('/', methods=['GET', 'POST'])
def index():
    crop_prediction = None
    fertilizer_prediction = None

    form_data = {
        'N': '', 'P': '', 'K': '', 'temperature': '',
        'humidity': '', 'ph': '', 'rainfall': '', 'moisture': '',
        'soil_type': '', 'crop_type': ''
    }

    if request.method == 'POST':
        form_type = request.form.get('form_type')

        try:
            if form_type == 'crop':
                N = float(request.form['N'])
                P = float(request.form['P'])
                K = float(request.form['K'])
                temperature = float(request.form['temperature'])
                humidity = float(request.form['humidity'])
                ph = float(request.form['ph'])
                rainfall = float(request.form['rainfall'])

                form_data.update({
                    'N': N, 'P': P, 'K': K,
                    'temperature': temperature, 'humidity': humidity,
                    'ph': ph, 'rainfall': rainfall
                })

                crop_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
                scaled_crop_features = crop_scaler.transform(crop_features)
                crop_result = crop_model.predict(scaled_crop_features)
                crop_prediction = crop_dict.get(int(crop_result[0]), "Unknown")

            elif form_type == 'fertilizer':
                temperature = float(request.form['temperature'])
                humidity = float(request.form['humidity'])
                N = float(request.form['N'])
                P = float(request.form['P'])
                K = float(request.form['K'])
                moisture = float(request.form['moisture'])
                soil_type = float(request.form['soil_type'])
                crop_type = float(request.form['crop_type'])

                form_data.update({
                    'moisture': moisture, 'soil_type': soil_type, 'crop_type': crop_type,
                    'temperature': temperature, 'humidity': humidity, 'N': N, 'P': P, 'K': K
                })

                fert_features = np.array([[temperature, humidity, moisture, soil_type, crop_type, N, P, K]])
                scaled_fert_features = fertilizer_scaler.transform(fert_features)
                fert_result = fertilizer_model.predict(scaled_fert_features)
                fertilizer_prediction = fertilizer_dict.get(int(fert_result[0]), "Unknown")

            log_prediction(crop_prediction, fertilizer_prediction)

        except Exception as e:
            crop_prediction = fertilizer_prediction = f"Error: {str(e)}"

    return render_template('index.html',
                           crop_prediction=crop_prediction,
                           fertilizer_prediction=fertilizer_prediction,
                           form_data=form_data)

from waitress import serve

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    serve(app, host='0.0.0.0', port=port)
