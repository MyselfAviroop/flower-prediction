import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Iris Flower Predictor", layout="centered")

# Load dataset
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df, iris.target_names

df, target_names = load_data()

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(df.iloc[:, :-1], df['species'])

# Sidebar Inputs
st.sidebar.header("🌼 Input Features")
sepal_length = st.sidebar.slider("Sepal Length (cm)", float(df["sepal length (cm)"].min()), float(df["sepal length (cm)"].max()), float(df["sepal length (cm)"].mean()))
sepal_width = st.sidebar.slider("Sepal Width (cm)", float(df["sepal width (cm)"].min()), float(df["sepal width (cm)"].max()), float(df["sepal width (cm)"].mean()))
petal_length = st.sidebar.slider("Petal Length (cm)", float(df["petal length (cm)"].min()), float(df["petal length (cm)"].max()), float(df["petal length (cm)"].mean()))
petal_width = st.sidebar.slider("Petal Width (cm)", float(df["petal width (cm)"].min()), float(df["petal width (cm)"].max()), float(df["petal width (cm)"].mean()))

# Main App
st.title("🌸 Iris Flower Species Prediction")
st.markdown("Enter the flower measurements on the left to predict the species using a trained Random Forest model.")

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)
predicted_species = target_names[prediction[0]]
prediction_proba = model.predict_proba(input_data)

# Prediction Output
st.subheader("🔍 Prediction Result")
st.success(f"Predicted Species: **{predicted_species}**")
st.write("Prediction Confidence (Probabilities):")
proba_df = pd.DataFrame(prediction_proba, columns=target_names)
st.dataframe(proba_df.style.highlight_max(axis=1, color='lightgreen'), use_container_width=True)

# Feature Importance
st.subheader("📊 Feature Importance")
feature_importance = pd.Series(model.feature_importances_, index=df.columns[:-1])
fig, ax = plt.subplots()
sns.barplot(x=feature_importance.values, y=feature_importance.index, palette="viridis", ax=ax)
ax.set_title("Feature Importance (Random Forest)")
st.pyplot(fig)

# Model Accuracy (Just for display)
st.sidebar.markdown("---")
st.sidebar.markdown("**Model Accuracy on Training Data:**")
st.sidebar.info(f"{model.score(df.iloc[:, :-1], df['species']) * 100:.2f}%")

