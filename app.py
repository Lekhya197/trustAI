import streamlit as st
import joblib
import re

@st.cache_resource
def load_tfidf():
    return joblib.load("tfidf.joblib")

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
