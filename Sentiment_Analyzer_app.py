from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model3
with open('model3.pkl', 'rb') as file:
    model3 = pickle.load(file)

# Load bow_counts for model3
with open('bow_counts_model3.pkl', 'rb') as file:
    bow_counts_model3 = pickle.load(file)


@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    tweet = ""

    if request.method == 'POST':
        tweet = request.form['tweet']
        if tweet:
            user_input_bow = bow_counts_model3.transform([tweet])
            prediction = model3.predict(user_input_bow)[0]

    return render_template('index.html', prediction=prediction, tweet=tweet)


if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')
