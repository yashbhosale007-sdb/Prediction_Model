import streamlit as st
import pandas as pd
import pickle
import numpy as np
import requests
from streamlit_lottie import st_lottie

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Impact Predictor", page_icon="🤖", layout="centered")

# --- CUSTOM CSS FOR ANIMATION & STYLE ---
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ASSETS ---
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_ai = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_m9p8llbg.json")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# --- APP HEADER ---
st_lottie(lottie_ai, height=200, key="coding")
st.title("🤖 AI Usage & Impact Predictor")
st.write("Enter student details below to predict the classification based on AI usage patterns.")

# --- INPUT FORM ---
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=10, max_value=100, value=20)
        gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
        education = st.selectbox("Education Level", options=["High School", "Undergraduate", "Postgraduate"])
        city = st.selectbox("City", options=["Metropolitan", "Urban", "Rural"]) # Update based on your data

    with col2:
        ai_tool = st.selectbox("AI Tool Used", options=["ChatGPT", "Gemini", "Claude", "Other"])
        usage_hours = st.slider("Daily Usage Hours", 0.0, 24.0, 2.0)
        purpose = st.selectbox("Purpose", options=["Academic", "Personal", "Work"])
        impact = st.selectbox("Impact on Grades", options=["Positive", "Neutral", "Negative"])

    submit = st.form_submit_button("Predict Result")

# --- PRE-PROCESSING & PREDICTION ---
# Note: KNN requires numeric inputs. Map your strings to the numbers used during training.
if submit:
    # Example Mapping (Adjust to match your LabelEncoder/OneHotEncoder)
    mapping = {
        "Gender": {"Male": 0, "Female": 1, "Other": 2},
        "Education": {"High School": 0, "Undergraduate": 1, "Postgraduate": 2},
        "City": {"Metropolitan": 0, "Urban": 1, "Rural": 2},
        "AI_Tool": {"ChatGPT": 0, "Gemini": 1, "Claude": 2, "Other": 3},
        "Purpose": {"Academic": 0, "Personal": 1, "Work": 2},
        "Impact": {"Positive": 0, "Neutral": 1, "Negative": 2}
    }

    features = np.array([[
        age, 
        mapping["Gender"][gender],
        mapping["Education"][education],
        mapping["City"][city],
        mapping["AI_Tool"][ai_tool],
        usage_hours,
        mapping["Purpose"][purpose],
        mapping["Impact"][impact]
    ]])

    prediction = model.predict(features)
    
    st.balloons()
    st.success(f"### Prediction Result: {prediction[0]}")
    
    with st.expander("See Input Summary"):
        st.write(f"Analyzing data for a {age} year old using AI for {usage_hours} hours daily...")
