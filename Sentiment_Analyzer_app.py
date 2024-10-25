import streamlit as st
import pickle
import nltk

# Download NLTK punkt tokenizer
nltk.download('punkt', quiet=True)

# Load model3
with open('model3.pkl', 'rb') as file:
    model3 = pickle.load(file)

# Load bow_counts for model3
with open('bow_counts_model3.pkl', 'rb') as file:
    bow_counts_model3 = pickle.load(file)

# Streamlit app title
st.title("Sentiment Analysis using Logistic")

# Add custom CSS for styling
st.markdown(
    """
    <style>
    /* Style for the button */
    .styled-button {
        background-color: orange; /* Button color */
        border: 2px solid blue; /* Button border color */
        color: white;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
        border-radius: 5px;
        transition: all 0.3s ease;
    }

    .styled-button:hover {
        background-color: blue; /* Change color on hover */
    }

    /* Center the title */
    h1 {
        text-align: center;
    }

    /* Style for the text area */
    textarea {
        display: block;
        margin: 0 auto;
        width: 80%;
        border: 2px solid blue; /* Text area border color */
        border-radius: 5px;
        padding: 10px;
        font-size: 16px;
    }

    /* Center the button container */
    .button-container {
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# User input for prediction
user_input = st.text_area("Enter a tweet to classify:")

# Create a button to analyze the input
if st.button("Analyze"):
    # Check if user input is provided
    if user_input.strip() == "":
        st.warning("Please enter a tweet to classify.")
    else:
        # Transform the user input using bow_counts_model3
        user_input_bow = bow_counts_model3.transform([user_input])

        # Prediction
        prediction = model3.predict(user_input_bow)[0]

        # Display the prediction result
        st.write(f"The sentiment of the input is: **{prediction}**")

        # Feedback section
        feedback = st.radio("Do you agree with this sentiment?", ("Yes", "No"))
        if st.button("Submit Feedback"):
            # Store the feedback (e.g., in a file or database)
            # For demonstration, just showing a message
            st.success(f"Feedback received: {feedback}")
