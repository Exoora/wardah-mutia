import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import string

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))

# Preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenize text
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]  # Remove stopwords
    return ' '.join(tokens)

# Load the TF-IDF Vectorizer and Model
@st.cache_resource
def load_vectorizer_model():
    with open('tfidf_vectorizer.pkl', 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)
    with open('multinomial_nb_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return vectorizer, model

vectorizer, model = load_vectorizer_model()

# Streamlit App Interface
st.title("Sentimen Analisis untuk Review")
st.subheader("Masukkan ulasan Anda untuk menganalisis sentimen (positif/negatif):")

# Input text area
user_input = st.text_area("Tulis ulasan Anda di sini:", placeholder="Contoh: Produk ini sangat bagus dan bermanfaat!")

# Analyze button
if st.button("Analisis Sentimen"):
    if user_input.strip() == "":
        st.error("Mohon masukkan ulasan untuk analisis.")
    else:
        # Preprocess and predict
        processed_input = preprocess_text(user_input)
        transformed_input = vectorizer.transform([processed_input])
        prediction = model.predict(transformed_input)[0]
        st.subheader(f"Sentimen: **{prediction.capitalize()}**")
