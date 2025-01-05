from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
import os
import json
from datetime import datetime

MODEL_INFO_FILE = 'model_info.json'

def fetch_nba_data():
    """Fetch NBA game data for the current season."""
    try:
        gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable='2023-24')
        games = gamefinder.get_data_frames()[0]
        return games
    except Exception as e:
        print(f"Error fetching NBA data: {str(e)}")
        return None

def create_game_pairs(games_df):
    """Create pairs of games (home and away) based on GAME_ID."""
    game_pairs = {}
    
    for _, game in games_df.iterrows():
        game_id = game['GAME_ID']
        if game_id not in game_pairs:
            game_pairs[game_id] = {'home': None, 'away': None}
            
        if '@' in game['MATCHUP']:
            game_pairs[game_id]['away'] = game
        else:
            game_pairs[game_id]['home'] = game
            
    return game_pairs

def preprocess_data(games_df):
    """Preprocess the NBA game data for model training."""
    if games_df is None:
        return None, None

    try:
        # Create game pairs
        game_pairs = create_game_pairs(games_df)
        
        # Create team mapping
        teams = sorted(games_df['TEAM_NAME'].unique())
        team_to_id = {team: idx for idx, team in enumerate(teams)}
        
        # Save team mapping
        with open('team_mapping.pkl', 'wb') as f:
            pickle.dump(team_to_id, f)

        # Prepare training data
        X = []
        y = []

        # Process complete game pairs
        for game_id, pair in game_pairs.items():
            if pair['home'] is not None and pair['away'] is not None:
                home_team = pair['home']['TEAM_NAME']
                away_team = pair['away']['TEAM_NAME']
                
                if home_team in team_to_id and away_team in team_to_id:
                    X.append([
                        team_to_id[home_team],
                        team_to_id[away_team]
                    ])
                    y.append([
                        pair['home']['PTS'],  # Home team points
                        pair['away']['PTS']   # Away team points
                    ])

        if len(X) == 0:
            print("No valid game pairs found")
            return None, None

        print(f"Found {len(X)} valid game pairs for training")
        return X, y
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        return None, None

def train_model(X, y):
    """Train the prediction model."""
    if X is None or y is None or len(X) == 0 or len(y) == 0:
        print("No training data available")
        return None

    try:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Save the model
        with open('nba_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        # Save model info
        model_info = {
            'last_updated': datetime.now().isoformat(),
            'num_games': len(X),
            'version': '1.0'
        }
        
        with open(MODEL_INFO_FILE, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        return model
    except Exception as e:
        print(f"Error training model: {str(e)}")
        return None

def should_retrain():
    """Check if model should be retrained based on last update time."""
    try:
        if not os.path.exists(MODEL_INFO_FILE):
            return True
            
        with open(MODEL_INFO_FILE, 'r') as f:
            model_info = json.load(f)
            
        last_updated = datetime.fromisoformat(model_info['last_updated'])
        hours_since_update = (datetime.now() - last_updated).total_seconds() / 3600
        
        # Retrain if more than 24 hours have passed
        return hours_since_update > 24
    except Exception:
        return True

def main():
    # Check if we should retrain
    if not should_retrain():
        print("Model is up to date. Last trained:", end=" ")
        with open(MODEL_INFO_FILE, 'r') as f:
            model_info = json.load(f)
            print(model_info['last_updated'])
        return

    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Fetch and save data
    print("Fetching NBA data...")
    games_df = fetch_nba_data()
    if games_df is not None:
        print("Saving data to CSV...")
        games_df.to_csv('data/nba_games.csv', index=False)
        
        # Preprocess data and train model
        print("Preprocessing data...")
        X, y = preprocess_data(games_df)
        if X and y:
            print("Training model...")
            model = train_model(X, y)
            if model:
                print("Model training completed successfully!")
            else:
                print("Failed to train model.")
        else:
            print("Failed to preprocess data.")
    else:
        print("Failed to fetch NBA data.")

if __name__ == "__main__":
    main() 