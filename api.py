from flask import Flask, request, jsonify
import numpy as np
import xgboost
import pickle

app = Flask(__name__)

with open('utils/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('utils/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def home_page():
    return 'Welcome to the credit scoring API'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        test_data = np.array(data['test_data'])
        scaled_data = scaler.transform(test_data)
        prediction = model.predict_proba(scaled_data)[:, 1]  # Assuming a binary classification task

        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

