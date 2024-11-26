import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Ensure NLTK resources are available
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('indonesian'))

# Load the trained model and TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

with open('wardah_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenize text
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]  # Remove stopwords
    return ' '.join(tokens)  # Join tokens back into a single string

# Generate word clouds
def generate_wordcloud(text, title, bgcolor='white'):
    wordcloud = WordCloud(width=800, height=400, background_color=bgcolor).generate(text)
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Streamlit app
def main():
    st.title("Analisis Sentimen Review Produk")
    st.write("Aplikasi ini digunakan untuk menganalisis sentimen review produk berdasarkan rating dan teks ulasan.")

    # Input for single review analysis
    st.header("Analisis Sentimen Individual")
    user_input = st.text_area("Masukkan ulasan Anda di sini:")
    
    if st.button("Analisis Sentimen"):
        if user_input:
            processed_input = preprocess_text(user_input)
            vectorized_input = tfidf_vectorizer.transform([processed_input])
            sentiment = model.predict(vectorized_input)[0]
            st.write(f"**Sentimen:** {sentiment.capitalize()}")
        else:
            st.warning("Silakan masukkan ulasan terlebih dahulu.")

    # Upload dataset for batch processing
    st.header("Analisis Sentimen Batch")
    uploaded_file = st.file_uploader("Unggah file CSV dengan kolom 'review'", type=['csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if 'review' in df.columns:
                df['cleaned_review'] = df['review'].apply(preprocess_text)
                df['sentiment'] = model.predict(tfidf_vectorizer.transform(df['cleaned_review']))
                df['sentiment'] = df['sentiment'].str.capitalize()
                st.write("Hasil Analisis Sentimen:")
                st.dataframe(df[['review', 'sentiment']])
                
                # Downloadable CSV
                csv_result = df[['review', 'sentiment']].to_csv(index=False).encode('utf-8')
                st.download_button(label="Unduh Hasil Analisis", data=csv_result, file_name="sentimen_hasil.csv", mime="text/csv")
            else:
                st.error("File harus memiliki kolom 'review'.")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}")

    # Visualizations
    st.header("Visualisasi WordCloud")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("WordCloud Sentimen Positif")
        positive_reviews = ' '.join(df[df['sentiment'] == 'Positive']['cleaned_review']) if 'sentiment' in df else ""
        if positive_reviews:
            generate_wordcloud(positive_reviews, "WordCloud Sentimen Positif", bgcolor='white')

    with col2:
        st.subheader("WordCloud Sentimen Negatif")
        negative_reviews = ' '.join(df[df['sentiment'] == 'Negative']['cleaned_review']) if 'sentiment' in df else ""
        if negative_reviews:
            generate_wordcloud(negative_reviews, "WordCloud Sentimen Negatif", bgcolor='black')

if __name__ == "__main__":
    main()
