import os
# --- THE FIX: specific to your computer's security settings ---
# This stops Python from trying to write the hidden log file that causes the crash.
if 'SSLKEYLOGFILE' in os.environ:
    del os.environ['SSLKEYLOGFILE']

import requests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# ... (Rest of your code stays exactly the same)

# --- STEP 1: Get Live Data from NHL API ---
def get_latest_stats():
    print("Fetching live NHL standings...")
    # This is the official endpoint used by the NHL website
    url = "https://api-web.nhle.com/v1/standings/now"
    response = requests.get(url)
    data = response.json()
    
    # Parse the complicated JSON into a simple list
    teams_data = []
    for team in data['standings']:
        teams_data.append({
            'Team': team['teamAbbrev']['default'],
            'Wins': team['wins'],
            'Losses': team['losses'],
            'Points': team['points'],
            'GoalDiff': team['goalDifferential'],
            'HomeWins': team['homeWins'],
            'RoadWins': team['roadWins']
        })
        
    return pd.DataFrame(teams_data)

# --- STEP 2: Pre-Train a "Logic" Model ---
# Since we can't download history easily, we will train a model on the
# *current* standings logic. Basically: "Better stats usually win."
# In a full version, you'd load 10 years of CSV history here.
def train_model():
    # We generate "synthetic matchups" based on current stats to teach the AI
    # that "High Points" + "High Goal Diff" = Winner.
    
    # 0 = Lose, 1 = Win
    # This is a simplified logic to get you started without a 1GB database
    X = []
    y = []
    
    # Teach the AI: If Team A has 20 more points than Team B, Team A wins.
    for _ in range(1000):
        points_diff = np.random.randint(-50, 50)
        goal_diff_diff = np.random.randint(-30, 30)
        
        # The Logic: Higher points & better goal diff = likely win
        score = (points_diff * 0.1) + (goal_diff_diff * 0.05)
        win_prob = 1 / (1 + np.exp(-score)) # Sigmoid function
        
        result = 1 if np.random.random() < win_prob else 0
        
        X.append([points_diff, goal_diff_diff])
        y.append(result)
        
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    return model

# --- STEP 3: The Prediction Function ---
def predict_game(team_home, team_away, df, model):
    # Find the stats for both teams
    try:
        home_stats = df[df['Team'] == team_home].iloc[0]
        away_stats = df[df['Team'] == team_away].iloc[0]
    except IndexError:
        print(f"Error: Could not find team {team_home} or {team_away}. Check spelling (e.g., COL, BOS, NYR).")
        return

    print(f"\n--- MATCHUP: {team_away} @ {team_home} ---")
    print(f"{team_home}: {home_stats['Points']} pts, {home_stats['GoalDiff']} GD")
    print(f"{team_away}: {away_stats['Points']} pts, {away_stats['GoalDiff']} GD")
    
    # Calculate the difference (Home - Away)
    diff_points = home_stats['Points'] - away_stats['Points']
    diff_goals = home_stats['GoalDiff'] - away_stats['GoalDiff']
    
    # Predict
    prob = model.predict_proba([[diff_points, diff_goals]])[0][1]
    
    print(f"\nPrediction Confidence for {team_home} win: {prob*100:.1f}%")
    if prob > 0.55:
        print(f"🏆 PREDICTION: {team_home} Wins!")
    elif prob < 0.45:
        print(f"🏆 PREDICTION: {team_away} Wins!")
    else:
        print("⚖️ PREDICTION: Too close to call (Overtime likely)")

# --- MAIN EXECUTION ---
df = get_latest_stats()
model = train_model()

# Example: Predict a game for your team
# You can change these to any 3-letter code: COL, BOS, TOR, MTL, etc.
predict_game('COL', 'BOS', df, model)
predict_game('NYR', 'NJD', df, model)

# --- MAIN EXECUTION ---
df = get_latest_stats()
model = train_model()

# List all valid team codes so the user knows what to type
valid_teams = sorted(df['Team'].unique())
print(f"\n✅ Valid Teams: {', '.join(valid_teams)}")

while True:
    print("\n--- 🏒 NHL PREDICTOR 🏒 ---")
    print("(Type 'Q' to quit)")
    
    # Ask user for input
    home_team = input("Enter HOME Team (e.g. COL): ").upper()
    if home_team == 'Q':
        break
        
    away_team = input("Enter AWAY Team (e.g. BOS): ").upper()
    if away_team == 'Q':
        break
        
    # Run the prediction
    if home_team in valid_teams and away_team in valid_teams:
        predict_game(home_team, away_team, df, model)
    else:
        print("❌ Error: Invalid team code. Please check the list above.")