from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# Load your model and vectorizer
with open('model3.pkl', 'rb') as file:
    model3 = pickle.load(file)

with open('bow_counts_model3.pkl', 'rb') as file:
    bow_counts_model3 = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    tweet = request.form['tweet']
    user_input_bow = bow_counts_model3.transform([tweet])
    prediction = model3.predict(user_input_bow)[0]
    return render_template('index.html', tweet=tweet, prediction=prediction)

@app.route('/react', methods=['POST'])
def react():
    data = request.get_json()
    reaction = data['reaction']
    # Here you can save the reaction to a database or file
    print(f"User reaction: {reaction}")  # For demonstration purposes
    return jsonify(reaction=reaction)

if __name__ == '__main__':
    app.run(debug=True)
