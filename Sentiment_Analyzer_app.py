import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Function to ensure NLTK resources are available
def ensure_nltk_resources():
    # Check for 'punkt' tokenizer
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    # Check for 'stopwords' corpus
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

# Call the function to ensure resources are available
ensure_nltk_resources()

# Load stopwords
stop_words = set(stopwords.words('english'))

# Load model3
with open('model3.pkl', 'rb') as file:
    model3 = pickle.load(file)

# Load bow_counts for model3
with open('bow_counts_model3.pkl', 'rb') as file:
    bow_counts_model3 = pickle.load(file)

# Streamlit app title
st.title("Sentiment Analysis using Logistic Regression")

# User input for prediction
user_input = st.text_area("Enter a tweet to classify:")

if st.button("Analyze"):
    if user_input:
        # Preprocess the user input
        # Tokenize
        tokens = word_tokenize(user_input.lower())
        # Remove stopwords
        filtered_tokens = [word for word in tokens if word not in stop_words]
        # Join tokens back into a string
        preprocessed_input = ' '.join(filtered_tokens)

        # Transform the preprocessed user input using bow_counts_model3
        user_input_bow = bow_counts_model3.transform([preprocessed_input])

        # Prediction
        prediction = model3.predict(user_input_bow)[0]

        # Display the prediction result
        st.write(f"The sentiment of the input is: {prediction}")
    else:
        st.write("Please enter a tweet to classify.")

# Optional: Add feedback feature
feedback = st.text_input("Provide feedback on the prediction (optional):")
if st.button("Submit Feedback"):
    if feedback:
        st.write("Thank you for your feedback!")
    else:
        st.write("No feedback provided.")
