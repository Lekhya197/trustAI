import streamlit as st
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model and vectorizer
@st.cache_resource  # Cache for performance
def load_artifacts():
    tfidf = joblib.load("tfidf.joblib")
    model = joblib.load("model.joblib")
    return tfidf, model

tfidf, model = load_artifacts()

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove special chars
    return text

# Streamlit UI
st.title("🛍️ Product Review Trust Analyzer")

# Text input and button
review = st.text_area("Paste a product review:", height=150)
analyze_clicked = st.button("🔍 Analyze Review")

if analyze_clicked and review:
    try:
        with st.spinner("Analyzing..."):
            # Preprocess and predict
            cleaned = clean_text(review)
            X = tfidf.transform([cleaned])
            is_fake = model.predict(X)[0]
            proba = model.predict_proba(X)[0][1]  # Probability of being fake
            
            # Show results
            st.success("Analysis Complete!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Prediction", 
                         "⚠️ Fake Review" if is_fake else "✅ Genuine Review",
                         f"{proba*100:.1f}% confidence")
                
            with col2:
                st.progress(proba if is_fake else 1-proba)
                
            # Explanation box
            with st.expander("📊 Analysis Details"):
                st.markdown(f"""
                - **Cleaned Text:** `{cleaned[:200]}...`
                - **Fake Probability:** {proba*100:.1f}%
                - **Key Words:** {', '.join(get_suspicious_words(cleaned, tfidf, model))}
                """)
                
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
elif analyze_clicked:
    st.warning("Please enter a review first!")

# Helper function to extract suspicious words (optional)
def get_suspicious_words(text, tfidf, model, top_n=5):
    words = tfidf.get_feature_names_out()
    X_vec = tfidf.transform([text])
    coefs = model.coef_[0]
    important_indices = X_vec.nonzero()[1]
    top_words = sorted(zip(important_indices, coefs[important_indices]), 
                      key=lambda x: abs(x[1]), reverse=True)[:top_n]
    return [words[i] for i, _ in top_words]
