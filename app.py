import streamlit as st
import pickle
import pandas as pd

# Load the model and vectorizer
model_path = "naive_bayes_model.pkl"
vectorizer_path = "tfidf_vectorizer.sav"
with open(model_path, "rb") as model_file, open(vectorizer_path, "rb") as vec_file:
    model = pickle.load(model_file)
    vectorizer = pickle.load(vec_file)

# Streamlit UI
st.title("Product Review Sentiment Analysis")
st.write("Classify product reviews as Positive or Negative")

# User input
review_text = st.text_area("Enter a product review:")

if st.button("Classify"):
    if review_text:
        # Transform and predict
        review_vector = vectorizer.transform([review_text])
        prediction = model.predict(review_vector)
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        st.write(f"The review is **{sentiment}**.")
    else:
        st.write("Please enter a review to classify.")
