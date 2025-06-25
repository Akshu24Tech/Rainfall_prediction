# 🌧️ Rainfall Prediction using Machine Learning

A regression-based ML project that predicts **annual rainfall** based on monthly values using models like Random Forest and Linear Regression.

## 📂 Project Structure
```bash
rainfall-prediction-ml/
├── data/ # Raw and processed data
├── models/ # Saved model
├── src/ # Scripts for preprocessing, training, evaluation
├── app/ # Streamlit UI
├── notebooks/ # EDA notebooks
├── requirements.txt
└── README.md
```


## 🚀 How to Run

1. Clone the repo
2. Create a virtual environment and install requirements:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate
   pip install -r requirements.txt
Run preprocessing:

```bash
python src/data_preprocessing.py
```
Train the model:

```bash
python src/train_model.py
```
Evaluate the model:

```bash
python src/evaluate_model.py
```
Launch app:
```bash
streamlit run app/streamlit_app.py
```
📈 Algorithms Used
Linear Regression

Random Forest Regressor

📊 Evaluation Metrics
R² Score

MAE (Mean Absolute Error)

MSE (Mean Squared Error)

📦 Dataset
Rainfall in India (1901–2015)
