import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

RAW_DATA_PATH = "data/raw/rainfall in india 1901-2015.csv"

def load_data():
    return pd.read_csv(RAW_DATA_PATH)

def handle_missing_values(df):
    return df.fillna(df.mean(numeric_only=True))

def select_features(df):
    # Drop unnecessary columns
    df = df.drop(columns=['SUBDIVISION', 'STATE', 'DISTRICT'], errors='ignore')
    # We'll use all months as features
    feature_cols = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                    'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    X = df[feature_cols]
    y = df['ANNUAL']
    return X, y

def normalize_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

def save_processed_data(X_train, X_test, y_train, y_test):
    os.makedirs("data/processed", exist_ok=True)
    pd.DataFrame(X_train).to_csv("data/processed/X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv("data/processed/X_test.csv", index=False)
    pd.DataFrame(y_train).to_csv("data/processed/y_train.csv", index=False)
    pd.DataFrame(y_test).to_csv("data/processed/y_test.csv", index=False)

def main():
    df = load_data()
    df = handle_missing_values(df)
    X,y = select_features(df)
    X_scaled, scaler = normalize_features(X)
    X_train, X_test, y_train, y_test  =split_data(X_scaled, y)
    save_processed_data(X_train, X_test, y_train, y_test)
    
if __name__  == "__main__":
    main()    