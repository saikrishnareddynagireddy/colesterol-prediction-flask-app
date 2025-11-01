from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("model/cholesterol_model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['age']),
            float(request.form['height']),
            float(request.form['weight']),
            float(request.form['ap_hi']),
            float(request.form['ap_lo']),
            float(request.form['gluc']),
            float(request.form['smoke']),
            float(request.form['alco']),
            float(request.form['active'])
        ]

        scaled_features = scaler.transform([features])
        prediction = model.predict(scaled_features)[0]

        return render_template('index.html', prediction_text=f"Predicted Cholesterol Level: {prediction:.2f} mg/dL")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
