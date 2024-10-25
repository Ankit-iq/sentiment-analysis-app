import streamlit as st
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Function to check and download NLTK resources
def ensure_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

# Call the function to ensure resources are available
ensure_nltk_resources()

# Check if nltk data is already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load stopwords
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# Sidebar for theme selection
theme = st.sidebar.selectbox("Select Theme", ["Light", "Dark"])

# CSS for light theme
light_theme = """
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
.stTextInput > div > input {
    margin-bottom: 20px;
    border-radius: 5px;
    border: 1px solid #007acc; /* Blue border */
    background-color: #f7f7f7;
    color: black; /* Black font color */
}
.stTextInput:focus {
    border-color: #005f99; /* Darker blue */
    box-shadow: 0 0 5px rgba(0, 95, 153, 0.8); /* Focus effect */
}
.footer {
    text-align: center;
    color: #007acc; /* Blue footer */
}
</style>
"""

# CSS for dark theme
dark_theme = """
<style>
body {
    background-color: #333; /* Dark background */
    color: #f0f0f0; /* Light text */
    font-family: 'Arial', sans-serif;
}
.main {
    background-color: #444; /* Darker card background */
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.5);
}
h1, h2, h3 {
    color: #007acc; /* Blue color */
}
.stTextInput > div > input {
    margin-bottom: 20px;
    border-radius: 5px;
    border: 1px solid #007acc; /* Blue border */
    background-color: #555; /* Dark input background */
    color: white; /* White font color */
}
.stTextInput:focus {
    border-color: #005f99; /* Darker blue */
    box-shadow: 0 0 5px rgba(0, 95, 153, 0.8); /* Focus effect */
}
.footer {
    text-align: center;
    color: #007acc; /* Blue footer */
}
</style>
"""

# Apply the selected theme
if theme == "Light":
    st.markdown(light_theme, unsafe_allow_html=True)
else:
    st.markdown(dark_theme, unsafe_allow_html=True)

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
    ax.set_title('Sentiment Counts', color='black' if theme == "Light" else 'white')
    ax.set_xlabel('Sentiment Types', color='black' if theme == "Light" else 'white')
    ax.set_ylabel('Counts', color='black' if theme == "Light" else 'white')
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


    X_val_bow = bow_counts.transform(val_data["lower"])
    y_val_bow = val_data['type']
    val_pred = model.predict(X_val_bow)
    val_accuracy = accuracy_score(y_val_bow, val_pred) * 100


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
        ngram_accuracy = accuracy_score(reviews_test['type'], test_pred_ngram) * 100
        st.write(f"N-gram Test Data Accuracy: {ngram_accuracy:.2f}%")

# Footer
st.markdown("<div class='footer'>Made by Ankit Kumar Bhuyan, Rajesh Kumar Panda and Bishnu Prasad Panda</div>", unsafe_allow_html=True)
