import streamlit as st
import joblib
import pandas as pd
import re
import string
import numpy as np

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load the model and vectorizer
model = joblib.load("best_fake_review_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def predict_review(review):
    cleaned_review = clean_text(review)
    transformed_review = vectorizer.transform([cleaned_review])
    prediction = model.predict(transformed_review)[0]

    # Check if the model supports predict_proba
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(transformed_review)[0]
        confidence = np.max(probabilities)
    else:
        confidence = None

    return "Fake Review" if prediction == 0 else "Genuine Review", confidence

# Streamlit App
st.title("Fake Review Detection App")

st.write("Enter a review below to check if it's fake or genuine:")

review = st.text_area("Review", "")

if st.button("Predict"):
    if review.strip():
        prediction, confidence = predict_review(review)
        st.write(f"Prediction: {prediction}")
        st.write(f"Confidence: {confidence:.2%}")
    else:
        st.write("Please enter a review to predict.")