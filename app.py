import math
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template
import traceback
from xgb_ensemble import XGBEnsemble
from flask_cors import CORS

class XGBEnsemble:
    def __init__(self, model_borderline, model_native, threshold=0.3):
        self.model_borderline = model_borderline
        self.model_native = model_native
        self.threshold = threshold

    def predict_proba(self, X):
        proba_borderline = self.model_borderline.predict_proba(X)[:, 1]
        proba_native = self.model_native.predict_proba(X)[:, 1]
        proba_avg = (proba_borderline + proba_native) / 2
        return np.vstack([1 - proba_avg, proba_avg]).T

    def predict(self, X):
        #print(X)
        print(self.predict_proba(X))
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

app = Flask(__name__)
CORS(app)

# Load trained scaler and model
scaler = joblib.load("scaler.pkl")
model = joblib.load("final_model.pkl")

@app.route('/')
def index():
    return render_template('index.html')

def preprocess(input_json):
    try:
        amount = input_json['Amount']
        time = input_json['Time']
        
        # Scale Amount and Time together (since scaler was fitted on both)
        scaled_amount, scaled_time = scaler.transform([[amount, time]])[0]
        
        # Compute Hour from Time in seconds since midnight
        hour = (time % 86400) // 3600  # hour in 0-23
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)
        
        # Extract V1-V28
        v_features = [input_json[f"V{i}"] for i in range(1, 29)]
        
        # Construct feature vector exactly like training order
        features = np.array([scaled_time] + v_features+ [scaled_amount, hour, hour_sin, hour_cos]).reshape(1, -1)
        return features
    except Exception as e:
        print("Error in preprocessing:", e)
        raise

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = preprocess(data)
        prediction = model.predict(features)[0]
        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        print("Unexpected error:", traceback.format_exc())
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
