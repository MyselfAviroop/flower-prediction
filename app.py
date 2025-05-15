import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



st.title('Hello Streamlit')
st.write('This is a simple Streamlit app to demonstrate the use of Streamlit for data visualization and analysis.')


df= pd.DataFrame({
    'First column': [1, 2, 3, 4, 5],
    'Second column': [10, 20, 30, 40, 50],
})
st.write('here is the dataframe:')
st.write(df)



chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a', 'b', 'c']
)
st.line_chart(chart_data)
