import streamlit as st
import pandas as pd
import re
import nltk
import pickle
from sklearn.metrics import accuracy_score

# Ensure NLTK resources are available
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# Load models and CountVectorizer
with open('model1.pkl', 'rb') as file:
    model1 = pickle.load(file)

with open('model3.pkl', 'rb') as file:
    model3 = pickle.load(file)

with open('bow_counts_model1.pkl', 'rb') as file:
    bow_counts_model1 = pickle.load(file)

with open('bow_counts_model3.pkl', 'rb') as file:
    bow_counts_model3 = pickle.load(file)

# Sidebar for theme selection
theme = st.sidebar.selectbox("Select Theme", ["Light", "Dark"])

st.title('Sentiment Analysis on Twitter Data')
st.sidebar.header('Upload CSV Files')

@st.cache_data()
def load_data(training_csv, validation_csv):
    training_data = pd.read_csv(training_csv, header=None)
    validation_data = pd.read_csv(validation_csv, header=None)
    return training_data, validation_data

# File uploader for train and validation data
uploaded_train_file = st.sidebar.file_uploader("Upload Training CSV", type=["csv"])
uploaded_val_file = st.sidebar.file_uploader("Upload Validation CSV", type=["csv"])

if uploaded_train_file and uploaded_val_file:
    training_df, validation_df = load_data(uploaded_train_file, uploaded_val_file)
    training_df.columns = ['id', 'information', 'type', 'text']
    validation_df.columns = ['id', 'information', 'type', 'text']

    def preprocess_text(df):
        df = df.copy()
        df["lower"] = df.text.str.lower()
        df["lower"] = df["lower"].apply(lambda x: re.sub('[^A-Za-z0-9 ]+', ' ', str(x)))
        return df

    train_data = preprocess_text(training_df)
    val_data = preprocess_text(validation_df)

    # Tokenization and vectorization for the models
    X_val_bow_model1 = bow_counts_model1.transform(val_data["lower"])
    y_val_bow = val_data['type']
    val_pred_model1 = model1.predict(X_val_bow_model1)
    val_accuracy_model1 = accuracy_score(y_val_bow, val_pred_model1) * 100

    X_val_bow_model3 = bow_counts_model3.transform(val_data["lower"])
    val_pred_model3 = model3.predict(X_val_bow_model3)
    val_accuracy_model3 = accuracy_score(y_val_bow, val_pred_model3) * 100

    st.subheader('Validation Accuracy')
    st.write(f"Model 1 (Unigram) Accuracy: {val_accuracy_model1:.2f}%")
    st.write(f"Model 3 (N-gram) Accuracy: {val_accuracy_model3:.2f}%")

    # User input for prediction
    st.header('Try Sentiment Classification')
    user_input = st.text_input("Enter a tweet to classify:", key="tweet_input")

    if st.button("Analyze"):
        if user_input:
            # Preprocess user input
            user_input_preprocessed = re.sub('[^A-Za-z0-9 ]+', ' ', user_input.lower())
            user_input_bow_model1 = bow_counts_model1.transform([user_input_preprocessed])
            user_input_bow_model3 = bow_counts_model3.transform([user_input_preprocessed])

            # Predict sentiment using both models
            prediction_model1 = model1.predict(user_input_bow_model1)[0]
            prediction_model3 = model3.predict(user_input_bow_model3)[0]

            st.write(f"Model 1 (Unigram) predicts: **{prediction_model1}**")
            st.write(f"Model 3 (N-gram) predicts: **{prediction_model3}**")
        else:
            st.warning("Please enter a tweet to analyze.")

# Footer
st.markdown("<div class='footer'>Made by Ankit Kumar Bhuyan</div>", unsafe_allow_html=True)
