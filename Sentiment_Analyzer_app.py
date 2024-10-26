from flask import Flask, render_template, request
import pickle
import logging

app = Flask(__name__)

# Load model3
try:
    with open('model3.pkl', 'rb') as file:
        model3 = pickle.load(file)
except Exception as e:
    logging.error(f"Error loading model3: {e}")

# Load bow_counts for model3
try:
    with open('bow_counts_model3.pkl', 'rb') as file:
        bow_counts_model3 = pickle.load(file)
except Exception as e:
    logging.error(f"Error loading bow_counts_model3: {e}")


@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    tweet = ""

    if request.method == 'POST':
        tweet = request.form.get('tweet', '')
        if tweet:
            try:
                user_input_bow = bow_counts_model3.transform([tweet])
                prediction = model3.predict(user_input_bow)[0]
            except Exception as e:
                logging.error(f"Error predicting sentiment: {e}")
                prediction = "An error occurred during prediction."

    return render_template('index.html', prediction=prediction, tweet=tweet)


if __name__ == '__main__':
    app.run(debug=True)
