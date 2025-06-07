import streamlit as st
import joblib
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Custom vectorizer loader to handle IDF weights
def load_tfidf():
    vectorizer = TfidfVectorizer(decode_error="replace")
    vec_params = joblib.load("tfidf.joblib")
    vectorizer.__dict__.update(vec_params)
    return vectorizer

# Cache resources
@st.cache_resource
def load_model():
    return joblib.load("model.joblib"), load_tfidf()

model, tfidf = load_model()

# Text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

# Streamlit UI
st.title("🔍 Fake Review Detector")
review = st.text_area("Paste a review:", height=150)
analyze_clicked = st.button("Analyze")

if analyze_clicked:
    if review:
        try:
            cleaned = clean_text(review)
            X = tfidf.transform([cleaned])
            
            # Ensure vectorizer is fitted
            if not hasattr(tfidf, 'idf_'):
                st.error("Vectorizer not properly initialized!")
            else:
                is_fake = model.predict(X)[0]
                proba = model.predict_proba(X)[0][1]
                
                st.metric("Result", 
                         "⚠️ Fake" if is_fake else "✅ Genuine",
                         f"{proba*100:.1f}% confidence")
                st.progress(int(proba * 100))
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a review first!")
