import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load model
model = joblib.load("models/random_forest.joblib")

# Load scaler (OPTIONAL: If you saved a scaler, load it here)
# scaler = joblib.load("models/scaler.joblib")

def predict_rainfall(monthly_values: list):
    """
    Accepts a list of 12 monthly rainfall values [JAN to DEC]
    Returns predicted annual rainfall
    """
    assert len(monthly_values) == 12, "Provide 12 monthly values."
    
    # Reshape input
    input_data = np.array(monthly_values).reshape(1, -1)
    
    # (Optional) Apply saved scaler
    # input_data = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_data)
    return prediction[0]

# Example usage
if __name__ == "__main__":
    sample_input = [5.1, 7.2, 8.0, 12.3, 30.2, 110.0, 200.4, 180.3, 90.2, 40.0, 10.2, 4.5]
    predicted = predict_rainfall(sample_input)
    print(f"Predicted annual rainfall: {predicted:.2f} mm")
