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
    return 'welcome to the credit scoring api'


if __name__ == '__main__':
    app.run(debug=True)