import sys
import json
import pickle
import os
import numpy as np

def load_model_and_mapping():
    """Load the trained model and team mapping."""
    try:
        with open('nba_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('team_mapping.pkl', 'rb') as f:
            team_mapping = pickle.load(f)
        return model, team_mapping
    except Exception as e:
        print(json.dumps({
            "error": f"Failed to load model or team mapping: {str(e)}"
        }))
        sys.exit(1)

def calculate_confidence(model, home_team_encoded, away_team_encoded, home_score, away_score):
    """Calculate prediction confidence based on model's trees predictions."""
    # Get predictions from all trees in the forest
    predictions = np.array([tree.predict([[home_team_encoded, away_team_encoded]])[0] 
                          for tree in model.estimators_])
    
    # Calculate standard deviation of predictions
    std_home = np.std(predictions[:, 0])
    std_away = np.std(predictions[:, 1])
    
    # Calculate mean absolute difference between predictions
    score_diff_predictions = predictions[:, 0] - predictions[:, 1]
    actual_diff = home_score - away_score
    
    # If most trees agree on the winner, higher confidence
    winner_agreement = np.mean(np.sign(score_diff_predictions) == np.sign(actual_diff))
    
    # Combine factors into confidence score
    max_std = max(std_home, std_away)
    confidence = winner_agreement * (1 - min(max_std / 100, 0.5))  # Normalize and cap std impact
    
    return min(max(confidence, 0.1), 0.95)  # Ensure confidence is between 10% and 95%

def predict_game(home_team, away_team):
    """Predict the outcome of a game between two teams."""
    try:
        # Load model and mapping
        model, team_mapping = load_model_and_mapping()
        
        # Validate teams
        if home_team not in team_mapping or away_team not in team_mapping:
            print(json.dumps({
                "error": "Invalid team name(s)"
            }))
            sys.exit(1)
        
        # Encode teams
        home_team_encoded = team_mapping[home_team]
        away_team_encoded = team_mapping[away_team]
        
        # Make prediction
        prediction = model.predict([[home_team_encoded, away_team_encoded]])[0]
        
        # Round scores to integers
        home_score = round(prediction[0])
        away_score = round(prediction[1])
        
        # Calculate confidence
        confidence = calculate_confidence(
            model, home_team_encoded, away_team_encoded, home_score, away_score
        )
        
        # Determine winner
        winner = home_team if home_score > away_score else away_team
        
        # Return prediction
        result = {
            "home_score": home_score,
            "away_score": away_score,
            "winner": winner,
            "confidence": confidence
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({
            "error": f"Prediction error: {str(e)}"
        }))
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(json.dumps({
            "error": "Invalid number of arguments. Expected: home_team away_team"
        }))
        sys.exit(1)
        
    home_team = sys.argv[1]
    away_team = sys.argv[2]
    predict_game(home_team, away_team) 