import joblib
import streamlit as st
from nltk.corpus import stopwords
import nltk

# Ensure stopwords are downloaded
nltk.download('stopwords')

# Load the combined model and vectorizer pipeline
pipeline = joblib.load('model_with_vectorizer_pipeline.sav')

# CSS for centering the main content
st.markdown("""
    <style>
        .main-content {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
        }
        .center-content {
            text-align: center;
        }
        .justify-text {
            text-align: justify;
        }
    </style>
""", unsafe_allow_html=True)

# Begin main content div
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# Title of the page (Centered)
st.markdown("""
<h2 class="center-content">SENTIMEN ANALIS PRODUK SKINCARE</h2>
<h4 class="center-content">Oleh: MUTIA ZAHIRMA 21.11.41**</h4>
""", unsafe_allow_html=True)

# Adding an image
st.image('wowww.jpeg')

# End main content div
st.markdown('</div>', unsafe_allow_html=True)

# Dropdown for About the App
section = st.selectbox(
    'Pilih Kategori',
    (
        'Klik untuk Memilih kategori',
        '1. Tentang Aplikasi',
        '2. Cara Penggunaan',
        '3. Tentang Model',
    )
)

# Content based on dropdown choice
if section == '1. Tentang Aplikasi':
    st.markdown("""
    <div class="justify-text">
     Tentang Aplikasi

    Aplikasi web ini bertujuan untuk blabla.
    </div>
    """, unsafe_allow_html=True)

elif section == '2. Cara Penggunaan':
    st.markdown("""
     Cara Penggunaan

    1. Masukkan teks ke dalam kotak input.
    2. Klik tombol 'Hasil Deteksi' untuk melihat prediksi.
    3. Prediksi akan muncul di bawah tombol.
    """)

elif section == '3. Tentang Model':
    st.markdown("""
    <div class="justify-text">
     Tentang Model

    Model yang digunakan dalam aplikasi ini adalah model Naïve Bayes.
    </div>
    """, unsafe_allow_html=True)

# Text input for prediction
input_text = st.text_input('Masukkan ulasan produk')

# Button to perform prediction
if st.button('Hasil Deteksi'):
    if input_text:  # Ensure input is not empty
        # Predict directly using the pipeline
        prediction = pipeline.predict([input_text])[0]
        probability = pipeline.predict_proba([input_text])[0]
        
        # Display the prediction and probabilities
        st.write(f"Prediction: {prediction}")
        st.write(f"Probability: {probability}")
