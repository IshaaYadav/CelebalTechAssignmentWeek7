# app.py

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from model_utils import predict_species

st.set_page_config(page_title="Penguin Species Predictor", layout="centered")

st.title("🐧 Penguin Species Predictor")
st.write(
    """
    This app predicts the **species of a penguin** based on your input features.
    The model is trained on the [Palmer Penguins dataset](https://github.com/allisonhorst/palmerpenguins).
    """
)

# 📌 Sidebar for user input
st.sidebar.header("Input Features")

def user_input_features():
    island = st.sidebar.selectbox("Island", ['Biscoe', 'Dream', 'Torgersen'])
    sex = st.sidebar.selectbox("Sex", ['Male', 'Female'])
    bill_length_mm = st.sidebar.slider("Bill Length (mm)", 32.0, 60.0, 44.0)
    bill_depth_mm = st.sidebar.slider("Bill Depth (mm)", 13.0, 21.0, 17.0)
    flipper_length_mm = st.sidebar.slider("Flipper Length (mm)", 170.0, 230.0, 200.0)
    body_mass_g = st.sidebar.slider("Body Mass (g)", 2700.0, 6300.0, 4000.0)
    
    sex_encoded = 0 if sex == 'Male' else 1

    data = {
        'island': island,
        'sex': sex_encoded,
        'bill_length_mm': bill_length_mm,
        'bill_depth_mm': bill_depth_mm,
        'flipper_length_mm': flipper_length_mm,
        'body_mass_g': body_mass_g
    }
    return data

input_data = user_input_features()

# 📌 Predict on button click
if st.button("Predict Species"):
    prediction, confidence = predict_species(input_data)
    st.success(f"Predicted Species: **{prediction}**")
    st.info(f"Model Confidence: **{confidence}%**")

    # 📊 Visual feedback
    st.subheader("Visual Comparison")

    # Plot user input as bar chart
    feature_df = pd.DataFrame([input_data])
    feature_df = feature_df.drop(columns=['island', 'sex'])
    st.bar_chart(feature_df.T.rename(columns={0: 'Your Input'}))

    # Add sample plot from dataset for comparison
    df = sns.load_dataset("penguins").dropna()
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="bill_length_mm", y="bill_depth_mm", hue="species", palette="Set2", alpha=0.7)
    ax.scatter(input_data['bill_length_mm'], input_data['bill_depth_mm'], color="black", s=100, label="You")
    plt.title("Bill Length vs Bill Depth")
    plt.legend()
    st.pyplot(fig)
