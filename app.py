import streamlit as st
st.set_page_config(page_title="Fake Review Detector", page_icon="🕵️‍♂️")

import joblib
import re

# Load vectorizer and model with caching
@st.cache_resource
def load_tfidf():
    try:
        return joblib.load("tfidf.joblib")
    except Exception as e:
        st.error(f"❌ Failed to load vectorizer: {e}")
        return None

@st.cache_resource
def load_model():
    try:
        return joblib.load("model.joblib")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

# Preprocess input review
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

# Load once
model = load_model()
tfidf = load_tfidf()

# UI
st.title("🕵️‍♂️ Fake Review Detector")
st.markdown("Paste a customer review below and click **Analyze** to check if it’s likely to be fake or genuine.")

review = st.text_area("💬 Review Text", height=150)
analyze_clicked = st.button("Analyze")

# Logic on button click
if analyze_clicked:
    if not review.strip():
        st.warning("⚠️ Please enter a review first!")
    elif not tfidf or not model:
        st.error("❌ Model or vectorizer is not loaded properly. Please check your files.")
    else:
        try:
            cleaned = clean_text(review)
            X = tfidf.transform([cleaned])
            is_fake = model.predict(X)[0]
            proba = model.predict_proba(X)[0][1]

            result_text = "⚠️ **Fake Review**" if is_fake else "✅ **Genuine Review**"
            confidence = f"Confidence: `{proba*100:.1f}%`"
            st.markdown(f"### {result_text}")
            st.markdown(confidence)
            st.progress(int(proba * 100))

        except Exception as e:
            st.error(f"❌ Analysis failed: {e}")
