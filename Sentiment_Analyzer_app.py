import streamlit as st
import pandas as pd
import re
import nltk
import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Specify the directory containing the NLTK data
nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')

# Set the NLTK data path
nltk.data.path.append(nltk_data_dir)

# Check if the punkt tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    st.error("Punkt tokenizer not found. Please ensure NLTK data is included in the project.")

# Load stopwords
try:
    stop_words = nltk.corpus.stopwords.words('english')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_dir)
    stop_words = nltk.corpus.stopwords.words('english')

# CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #f0f0f0; /* Light background */
        color: #333; /* Dark text */
        font-family: 'Arial', sans-serif;
    }

    .main {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    }

    h1, h2, h3 {
        color: #007acc; /* Blue color */
    }

    .stTextInput {
        margin-bottom: 20px;
        border-radius: 5px;
        border: 1px solid #007acc; /* Blue border */
        background-color: #f7f7f7;
    }

    .stTextInput:focus {
        border-color: #005f99; /* Darker blue */
        box-shadow: 0 0 5px rgba(0, 95, 153, 0.8); /* Focus effect */
    }

    .stButton {
        background-color: #007acc; /* Blue button */
        color: white; /* White text */
        border-radius: 5px;
    }

    .stButton:hover {
        background-color: #005f99; /* Darker blue on hover */
    }

    footer {
        text-align: center;
        color: #007acc; /* Blue footer */
    }
    </style>
    """, unsafe_allow_html=True
)

st.title('Sentiment Analysis on Twitter Data')
st.sidebar.header('Upload CSV Files')

@st.cache_data()
def load_data(train_file, val_file):
    train = pd.read_csv(train_file, header=None)
    val = pd.read_csv(val_file, header=None)
    return train, val

# File uploader for train and validation data
train_file = st.sidebar.file_uploader("Upload Training CSV", type=["csv"])
val_file = st.sidebar.file_uploader("Upload Validation CSV", type=["csv"])

if train_file and val_file:
    train, val = load_data(train_file, val_file)
    train.columns = ['id', 'information', 'type', 'text']
    val.columns = ['id', 'information', 'type', 'text']

    def preprocess_text(df):
        df = df.copy()  # Create a copy to avoid modifying the original DataFrame
        df["lower"] = df.text.str.lower()
        df["lower"] = df["lower"].apply(lambda x: re.sub('[^A-Za-z0-9 ]+', ' ', str(x)))
        return df

    train_data = preprocess_text(train)
    val_data = preprocess_text(val)

    # Visualize sentiment distribution
    sentiment_counts = train_data['type'].value_counts()
    st.subheader('Sentiment Distribution in Training Data')
    fig, ax = plt.subplots()
    sentiment_counts.plot(kind='bar', ax=ax, color='blue', edgecolor='black')
    ax.set_title('Sentiment Counts', color='black')
    ax.set_xlabel('Sentiment Types', color='black')
    ax.set_ylabel('Counts', color='black')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Tokenize and prepare data for model
    bow_counts = CountVectorizer(stop_words=stop_words, ngram_range=(1, 1))
    reviews_train, reviews_test = train_test_split(train_data, test_size=0.2, random_state=0)
    X_train_bow = bow_counts.fit_transform(reviews_train["lower"])
    X_test_bow = bow_counts.transform(reviews_test["lower"])

    model = LogisticRegression(C=1, solver="liblinear", max_iter=1000)
    model.fit(X_train_bow, reviews_train['type'])

    test_pred = model.predict(X_test_bow)
    accuracy = accuracy_score(reviews_test['type'], test_pred) * 100
    st.write(f"Test Data Accuracy: {accuracy:.2f}%")

    X_val_bow = bow_counts.transform(val_data["lower"])
    y_val_bow = val_data['type']
    val_pred = model.predict(X_val_bow)
    val_accuracy = accuracy_score(y_val_bow, val_pred) * 100
    st.write(f"Validation Data Accuracy: {val_accuracy:.2f}%")

    # User input for prediction
    st.header('Try Sentiment Classification')
    user_input = st.text_input("Enter a tweet to classify:", key="tweet_input")

    if st.button("Analyze"):
        if user_input:
            user_input_preprocessed = re.sub('[^A-Za-z0-9 ]+', ' ', user_input.lower())
            user_input_bow = bow_counts.transform([user_input_preprocessed])

            # Predict sentiment of user input
            prediction = model.predict(user_input_bow)[0]
            st.write(f"The sentiment of the input tweet is: **{prediction}**")
        else:
            st.warning("Please enter a tweet to analyze.")

    # Optional: N-grams model
    if st.sidebar.checkbox('Use n-grams (up to 4)'):
        bow_counts = CountVectorizer(stop_words=stop_words, ngram_range=(1, 4))
        X_train_bow = bow_counts.fit_transform(reviews_train["lower"])
        X_test_bow = bow_counts.transform(reviews_test["lower"])
        X_val_bow = bow_counts.transform(val_data["lower"])

        model_ngram = LogisticRegression(C=0.9, solver="liblinear", max_iter=1500)
        model_ngram.fit(X_train_bow, reviews_train['type'])
        test_pred_ngram = model_ngram.predict(X_test_bow)
        accuracy_ngram = accuracy_score(reviews_test['type'], test_pred_ngram) * 100
        st.write(f"Test Data Accuracy with n-grams: {accuracy_ngram:.2f}%")
        val_pred_ngram = model_ngram.predict(X_val_bow)
        val_accuracy_ngram = accuracy_score(y_val_bow, val_pred_ngram) * 100
        st.write(f"Validation Data Accuracy with n-grams: {val_accuracy_ngram:.2f}%")

    # Display a footer
    st.markdown("<footer>Created by Ankit Kumar Bhuyan</footer>", unsafe_allow_html=True)
