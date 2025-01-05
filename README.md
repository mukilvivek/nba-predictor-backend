# NBA Predictor Backend

This is the backend service for the NBA Game Predictor application. It provides a REST API for predicting NBA game outcomes using machine learning.

## Features

- NBA game outcome predictions
- Machine learning model trained on historical NBA data
- REST API endpoints for predictions
- Health check endpoint

## API Endpoints

- `POST /predict`: Get game prediction
  - Request body: `{ "homeTeam": "Team Name", "roadTeam": "Team Name" }`
  - Response: `{ "winner": "Team Name", "score": "Score Prediction", "confidence": 85 }`

- `GET /health`: Health check endpoint
  - Response: `{ "status": "healthy", "model_version": "1.0.0" }`

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python app.py
```

The server will start on port 5000 by default.

## Deployment

This application is configured for deployment on Render.com using Gunicorn as the WSGI server.

## Model Updates

The model is automatically updated daily using the `update_model.py` script, which fetches the latest NBA game data and retrains the model. 