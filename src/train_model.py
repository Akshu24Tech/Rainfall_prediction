import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def load_processed_data():
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()  # Flatten for sklearn
    return X_train, y_train


def train_models(X_train, y_train):
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }

    trained_models = {}
    scores = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_train)
        r2 = r2_score(y_train, preds)
        mae = mean_absolute_error(y_train, preds)
        trained_models[name] = model
        scores[name] = {"R2": r2, "MAE": mae}
        print(f"{name} — R2: {r2:.3f}, MAE: {mae:.2f}")

    return trained_models, scores


def save_best_model(models, scores):
    best_model_name = max(scores, key=lambda name: scores[name]["R2"])
    best_model = models[best_model_name]
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, f"models/{best_model_name.replace(' ', '_').lower()}.joblib")
    print(f"Best model saved: {best_model_name}")
    
def main():
    X_train, y_train = load_processed_data()
    models, scores = train_models(X_train, y_train)
    save_best_model(models, scores)

if __name__ == "__main__":
    main()
    