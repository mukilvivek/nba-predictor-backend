from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
import os
from predict import make_prediction

app = Flask(__name__)
CORS(app)

# Load model and team mapping
model = joblib.load('nba_model.pkl')
team_mapping = joblib.load('team_mapping.pkl')
with open('model_info.json', 'r') as f:
    model_info = json.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        home_team = data.get('homeTeam')
        road_team = data.get('roadTeam')
        
        if not home_team or not road_team:
            return jsonify({'error': 'Both home and road teams are required'}), 400
            
        prediction = make_prediction(home_team, road_team, model, team_mapping)
        return jsonify(prediction)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_version': model_info.get('version', 'unknown')})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 