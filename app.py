import streamlit as st
import joblib
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Custom vectorizer loader that works reliably
@st.cache_resource
def load_tfidf():
    try:
        # Try loading the full vectorizer first
        return joblib.load("tfidf.joblib")
    except:
        # Fallback: Reconstruct vectorizer from components
        vec_params = joblib.load("tfidf.joblib")
        vectorizer = TfidfVectorizer()
        vectorizer.vocabulary_ = vec_params['vocabulary_']
        vectorizer.idf_ = vec_params['idf_']
        if 'stop_words_' in vec_params:
            vectorizer.stop_words_ = vec_params['stop_words_']
        return vectorizer

@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

model = load_model()
tfidf = load_tfidf()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

st.title("🔍 Fake Review Detector")
review = st.text_area("Paste a review:", height=150)
analyze_clicked = st.button("Analyze")

if analyze_clicked:
    if review:
        try:
            cleaned = clean_text(review)
            X = tfidf.transform([cleaned])
            
            if not hasattr(tfidf, 'vocabulary_'):
                st.error("Vectorizer not properly loaded!")
            else:
                is_fake = model.predict(X)[0]
                proba = model.predict_proba(X)[0][1]
                
                st.metric("Result", 
                         "⚠️ Fake" if is_fake else "✅ Genuine",
                         f"{proba*100:.1f}% confidence")
                st.progress(int(proba * 100))
                
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
    else:
        st.warning("Please enter a review first!")
