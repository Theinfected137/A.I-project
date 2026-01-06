from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

pipeline = joblib.load('iris_pipeline.pkl')

species_map = {
    0: 'Iris-setosa',
    1: 'Iris-versicolor',
    2: 'Iris-virginica'
}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # {"features": [5.9, 3.0, 5.1, 1.8]}
    features = np.array(data['features']).reshape(1, -1)

    pred = pipeline.predict(features)[0]

    return jsonify({'species': species_map[pred]})

if __name__ == '__main__':
    app.run(debug=True)
