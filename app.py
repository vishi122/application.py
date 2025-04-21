# app.py
from flask import Flask, request, jsonify
import requests
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

class CricketPredictor:
    def __init__(self, api_key):
        self.scraper_api_key = api_key
        self.model = LinearRegression()
        # Sample trained data (in a real app, you'd train this properly)
        self.model.coef_ = np.array([0.5, -0.3, 1.2, 0.8, -2.1, 3.4, 0.2, 0.7])
        self.model.intercept_ = 30
    
    def predict_score(self, data):
        features = np.array([
            data['team1_rating'],
            data['team2_rating'],
            data['venue_avg'],
            data['current_runs'],
            data['current_wickets'],
            data['overs_remaining'],
            data['is_day_night'],
            data['pitch_condition']
        ]).reshape(1, -1)
        
        prediction = self.model.predict(features)
        return int(prediction[0])

predictor = CricketPredictor("d631ecf396e4ded97f4f57a67f04e245")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        prediction = predictor.predict_score(data)
        return jsonify({
            'status': 'success',
            'prediction': prediction,
            'message': f"Predicted final score: {prediction}"
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/')
def home():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(debug=True)