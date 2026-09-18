import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Car Price Prediction Dashboard", layout="wide")

st.title("🚗 Car Price Prediction with Machine Learning")
st.markdown("A Machine Learning web application to train a regression model and predict car prices based on features like horsepower, mileage, and brand.")

# Safe file path loading
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'car data.csv')

@st.cache_data
def load_data():
    df = pd.read_csv(file_path)
    return df

try:
    df = load_data()
    
    # Dataset Preview Section
    if st.checkbox("Show Raw Dataset Preview"):
        st.subheader("Dataset Head")
        st.dataframe(df.head())

    # Missing values check
    if st.checkbox("Show Missing Values Info"):
        st.subheader("Missing Values Count")
        st.write(df.isnull().sum())

    # Preprocessing and Model Training
    # Copy dataset for processing
    df_processed = df.copy()
    le = LabelEncoder()
    for col in df_processed.select_dtypes(include=['object']).columns:
        df_processed[col] = le.fit_transform(df_processed[col])

    # Target Selection
    if 'Selling_Price' in df_processed.columns:
        X = df_processed.drop(columns=['Selling_Price'])
        y = df_processed['Selling_Price']
    elif 'Price' in df_processed.columns:
        X = df_processed.drop(columns=['Price'])
        y = df_processed['Price']
    else:
        X = df_processed.iloc[:, :-1]
        y = df_processed.iloc[:, -1]

    # Model Training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)

    # Evaluation Metrics Display
    st.subheader("📊 Model Evaluation Results")
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    col1, col2 = st.columns(2)
    col1.metric("Root Mean Squared Error (RMSE)", f"{rmse:.4f}")
    col2.metric("R2 Score", f"{r2:.4f}")

    # Visualization
    st.subheader("📈 Actual vs Predicted Car Prices")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=y_test, y=y_pred, color='blue', alpha=0.7, ax=ax)
    ax.set_xlabel("Actual Prices")
    ax.set_ylabel("Predicted Prices")
    ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
    st.pyplot(fig)

except Exception as e:
    st.error(f"Error loading data or running model: {e}")