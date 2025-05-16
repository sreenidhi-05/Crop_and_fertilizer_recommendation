from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load models and scalers
with open('crop_model.sav', 'rb') as f:
    crop_model = pickle.load(f)
with open('crop_scaler.sav', 'rb') as f:
    crop_scaler = pickle.load(f)
with open('fertilizer_model.sav', 'rb') as f:
    fertilizer_model = pickle.load(f)
with open('fertilizer_scaler.sav', 'rb') as f:
    fertilizer_scaler = pickle.load(f)

# Crop dictionary
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

@app.route('/', methods=['GET', 'POST'])
def index():
    crop_prediction = None
    fertilizer_prediction = None

    if request.method == 'POST':
        try:
            # Crop inputs
            N = float(request.form['N'])
            P = float(request.form['P'])
            K = float(request.form['K'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])

            crop_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            scaled_crop_features = crop_scaler.transform(crop_features)
            crop_result = crop_model.predict(scaled_crop_features)
            crop_prediction = crop_dict.get(int(crop_result[0]), "Unknown")

            # Fertilizer inputs
            moisture = float(request.form['moisture'])
            soil_type = float(request.form['soil_type'])
            crop_type = float(request.form['crop_type'])

            fert_features = np.array([[temperature, humidity, moisture, soil_type, crop_type, N, P, K]])
            scaled_fert_features = fertilizer_scaler.transform(fert_features)
            fert_result = fertilizer_model.predict(scaled_fert_features)
            fertilizer_prediction = fertilizer_dict.get(int(fert_result[0]), "Unknown")

        except Exception as e:
            crop_prediction = fertilizer_prediction = f"Error: {str(e)}"

    return render_template('index.html',
                           crop_prediction=crop_prediction,
                           fertilizer_prediction=fertilizer_prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
