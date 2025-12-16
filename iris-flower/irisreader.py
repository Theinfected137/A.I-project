from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model_path = r'C:\Users\4534\Documents\A.I_project\iris-flower\iris_model.pkl'
scaler_path = r'C:\Users\4534\Documents\A.I_project\iris-flower\scaler.pkl'

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Espera JSON: {"features": [5.9, 3.0, 5.1, 1.8]}
    features = np.array(data['features']).reshape(1, -1)
    scaled = scaler.transform(features)
    pred = model.predict(scaled)[0]
    species = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}[pred]
    return jsonify({'species': species})

if __name__ == '__main__':
    app.run(debug=True)