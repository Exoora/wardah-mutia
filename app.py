import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved model and vectorizer
model_filename = 'sentiment_analysis_model.pkl'
vectorizer_filename = 'tfidf_vectorizer.pkl'

model = joblib.load(model_filename)
vectorizer = joblib.load(vectorizer_filename)

# Title of the app
st.title("Sentiment Analysis App")
st.write("Enter a comment below to predict its sentiment (positive or negative).")

# Text input
user_input = st.text_area("Enter your comment here:")

# Predict button
if st.button("Predict"):
    if user_input.strip():
        # Preprocess and vectorize the input
        processed_input = vectorizer.transform([user_input])
        
        # Make prediction
        prediction = model.predict(processed_input)[0]
        prediction_proba = model.predict_proba(processed_input)[0]
        
        # Display the result
        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = max(prediction_proba) * 100
        st.subheader(f"Predicted Sentiment: {sentiment}")
        st.write(f"Confidence: {confidence:.2f}%")
    else:
        st.error("Please enter a valid comment.")
