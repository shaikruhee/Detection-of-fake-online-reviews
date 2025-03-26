!pip install joblib
import streamlit as st
import joblib
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the model and vectorizer
model = joblib.load("best_fake_review_model.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Streamlit app
def main():
    st.title("Fake Online Reviews Detection")
    st.write("Enter a review below to check if it's fake or real.")

    # Input text box
    review = st.text_area("Enter your review:")

    if st.button("Predict"):
        if review.strip():
            # Preprocess and predict
            cleaned_review = clean_text(review)
            vectorized_review = tfidf_vectorizer.transform([cleaned_review])
            prediction = model.predict(vectorized_review)[0]
            confidence = model.predict_proba(vectorized_review)[0][prediction] * 100


            # Show result
            if prediction == 0:
                st.error(f"FAKE Review. Confidence: {confidence:.2f}%")
            else:
                st.success(f"REAL review. Confidence: {confidence:.2f}%")
        else:
            st.warning("Please enter a review to analyze.")

if __name__ == "__main__":
    main()
