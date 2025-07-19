# model_utils.py

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

#Train the model (called once during development)
def train_and_save_model():
    df = sns.load_dataset("penguins").dropna()
    df['sex'] = df['sex'].map({'Male': 0, 'Female': 1})
    df = pd.get_dummies(df, columns=['island'], drop_first=True)

    X = df.drop('species', axis=1)
    y = df['species']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/penguin_model.pkl")
    return model

#Load model
def load_model():
    model_path = "models/penguin_model.pkl"
    if not os.path.exists(model_path):
        return train_and_save_model()
    return joblib.load(model_path)

#Preprocess input and predict
def predict_species(input_dict):
    model = load_model()
    df_input = pd.DataFrame([input_dict])

    # Manual one-hot encoding for island
    for island in ['island_Biscoe', 'island_Dream', 'island_Torgersen']:
        df_input[island] = 0
    island_col = f"island_{input_dict['island']}"
    if island_col in df_input.columns:
        df_input[island_col] = 1

    # Drop original columns
    df_input = df_input.drop(['island'], axis=1)

    # Reorder & fill missing columns if any
    model_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 
                      'sex', 'island_Dream', 'island_Torgersen']
    for col in model_features:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[model_features]
    
    prediction = model.predict(df_input)[0]
    prob = model.predict_proba(df_input).max()
    return prediction, round(prob * 100, 2)
