import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os

def load_test_data():
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()
    return X_test, y_test

def load_model(model_name="random_forest.joblib"):
    model_path = os.path.join("models", model_name)
    return joblib.load(model_path)

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    print(f"R² Score: {r2:.3f}")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")

    return predictions

def visualize_results(y_test, predictions):
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_test)), y_test, label='Actual', color='blue', alpha=0.6)
    plt.scatter(range(len(predictions)), predictions, label='Predicted', color='red', alpha=0.6)
    plt.title("Actual vs Predicted Rainfall")
    plt.xlabel("Sample Index")
    plt.ylabel("Rainfall (mm)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    X_test, y_test = load_test_data()
    model = load_model("random_forest.joblib")  # or 'linear_regression.joblib'
    predictions = evaluate_model(model, X_test, y_test)
    visualize_results(y_test, predictions)

if __name__ == "__main__":
    main()

