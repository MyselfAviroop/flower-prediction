import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load data and cache
@st.cache_data
def load_data():
    iris = load_iris()
    data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    data['species'] = iris.target
    return data, iris.target_names

# Load dataset
df, target_names = load_data()

# Train model
model = RandomForestClassifier()
model.fit(df.iloc[:, :-1], df['species'])

# Sidebar UI
st.sidebar.title("Input Parameters")
sepal_length = st.sidebar.slider("Sepal Length", float(df["sepal length (cm)"].min()), float(df["sepal length (cm)"].max()), float(df["sepal length (cm)"].mean()))
sepal_width = st.sidebar.slider("Sepal Width", float(df["sepal width (cm)"].min()), float(df["sepal width (cm)"].max()), float(df["sepal width (cm)"].mean()))
petal_length = st.sidebar.slider("Petal Length", float(df["petal length (cm)"].min()), float(df["petal length (cm)"].max()), float(df["petal length (cm)"].mean()))
petal_width = st.sidebar.slider("Petal Width", float(df["petal width (cm)"].min()), float(df["petal width (cm)"].max()), float(df["petal width (cm)"].mean()))

# Predict
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(input_data)
predicted_species = target_names[prediction[0]]

# Output
st.write("### Prediction")
st.write(f"The predicted species is: **{predicted_species}**")

