# app.py
import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load artifacts
tfidf = joblib.load("tfidf.joblib")
model = joblib.load("model.joblib")

# Preprocessing function (same as Colab)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

# Streamlit UI
st.title("🔍 Fake Review Detector")
review = st.text_area("Paste a review:")

if review:
    cleaned = clean_text(review)
    X = tfidf.transform([cleaned])
    pred = model.predict(X)[0]
    st.metric("Prediction", "FAKE 🚩" if pred == 1 else "GENUINE ✅")