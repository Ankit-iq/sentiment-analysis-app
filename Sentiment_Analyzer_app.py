import streamlit as st
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Check if nltk data is already downloaded
try:
    nltk.data.find('corpora/stopwords.zip')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt.zip')
except LookupError:
    nltk.download('punkt_tab')

# Load stopwords
stopwords_nltk = nltk.corpus.stopwords
stop_words = stopwords_nltk.words('english')

st.title('Sentiment Analysis on Textual Data')
@st.cache_data
def load_data(train_file, val_file):
    train = pd.read_csv(train_file, header=None)
    val = pd.read_csv(val_file, header=None)
    return train, val

# File uploader for train and validation data
st.sidebar.header('Upload CSV Files')
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

    tokens_text = [nltk.word_tokenize(str(word)) for word in train_data["lower"]]
    tokens_counter = [item for sublist in tokens_text for item in sublist]
    st.write("Number of Unique Tokens:", len(set(tokens_counter)))

    bow_counts = CountVectorizer(
        stop_words=stop_words,
        ngram_range=(1, 1)
    )

    reviews_train, reviews_test = train_test_split(train_data, test_size=0.2, random_state=0)
    X_train_bow = bow_counts.fit_transform(reviews_train["lower"])
    X_test_bow = bow_counts.transform(reviews_test["lower"])

    model1 = LogisticRegression(C=1, solver="liblinear", max_iter=1000)
    model1.fit(X_train_bow, reviews_train['type'])

    test_pred = model1.predict(X_test_bow)
    accuracy = accuracy_score(reviews_test['type'], test_pred) * 100
    st.write(f"Test Data Accuracy: {accuracy:.2f}%")

    X_val_bow = bow_counts.transform(val_data["lower"])
    y_val_bow = val_data['type']
    val_pred = model1.predict(X_val_bow)
    val_accuracy = accuracy_score(y_val_bow, val_pred) * 100
    st.write(f"Validation Data Accuracy: {val_accuracy:.2f}%")

    # User input for prediction
    st.header('Try Sentiment Classification')
    user_input = st.text_input("Enter a tweet to classify:")

    if st.button("Analyze"):
        if user_input:
            user_input_preprocessed = re.sub('[^A-Za-z0-9 ]+', ' ', user_input.lower())
            user_input_bow = bow_counts.transform([user_input_preprocessed])

            # Predict sentiment of user input
            prediction = model1.predict(user_input_bow)[0]
            st.write(f"The sentiment of the input tweet is: **{prediction}**")
        else:
            st.warning("Please enter a tweet to analyze.")

    # Optional: N-grams model
    if st.sidebar.checkbox('Use n-grams (up to 4)'):
        bow_counts = CountVectorizer(
            stop_words=stop_words,
            ngram_range=(1, 4)
        )

        X_train_bow = bow_counts.fit_transform(reviews_train["lower"])
        X_test_bow = bow_counts.transform(reviews_test["lower"])
        X_val_bow = bow_counts.transform(val_data["lower"])

        model3 = LogisticRegression(C=0.9, solver="liblinear", max_iter=1500)
        model3.fit(X_train_bow, reviews_train['type'])
        test_pred_3 = model3.predict(X_test_bow)
        accuracy_3 = accuracy_score(reviews_test['type'], test_pred_3) * 100
        st.write(f"Test Data Accuracy with n-grams: {accuracy_3:.2f}%")
        val_pred_3 = model3.predict(X_val_bow)
        val_accuracy_3 = accuracy_score(y_val_bow, val_pred_3) * 100
        st.write(f"Validation Data Accuracy with n-grams: {val_accuracy_3:.2f}%")
