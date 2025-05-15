import streamlit as st

st.title('🌟 Streamlit Input Widgets Showcase')

# Text input
name = st.text_input('Enter your name')
if name:
    st.write(f'Hello, {name}!')

# Text area
bio = st.text_area('Tell us about yourself')
if bio:
    st.write('Bio:', bio)

# Number input
age = st.number_input('Enter your age', min_value=0, max_value=120, value=25)
st.write(f'You are {age} years old.')

# Slider
rating = st.slider('Rate this app', 0, 10, 5)
st.write(f'You rated: {rating}/10')

# Checkbox
subscribe = st.checkbox('Subscribe to newsletter')
if subscribe:
    st.write('✅ Subscribed!')

# Radio buttons
gender = st.radio("Select your gender", ('Male', 'Female', 'Other'))
st.write(f'Gender: {gender}')

# Select box
country = st.selectbox("Choose your country", ['India', 'USA', 'UK', 'Canada', 'Other'])
st.write(f'Country: {country}')

# Multiselect
skills = st.multiselect("Select your skills", ['Python', 'JavaScript', 'C++', 'AI', 'ML'])
if skills:
    st.write('Your skills:', ', '.join(skills))

# Date input
dob = st.date_input('Select your date of birth')
st.write(f'Date of birth: {dob}')

# Time input
time = st.time_input('Pick a time')
st.write(f'Time selected: {time}')

# File uploader
uploaded_file = st.file_uploader("Upload a file")
if uploaded_file is not None:
    st.write("Uploaded file name:", uploaded_file.name)

# Color picker
color = st.color_picker('Pick a color', '#00f900')
st.write(f'You selected: {color}')
