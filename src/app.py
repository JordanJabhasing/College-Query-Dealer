import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import random

app = Flask(__name__)
CORS(app)

# Load model and vectorizer
model = pickle.load(open(r"D:\AI_Chatbot_Project\models\intent_model.pkl", 'rb'))
vectorizer = pickle.load(open(r"D:\AI_Chatbot_Project\models\vectorizer.pkl", 'rb'))

# Load cleaned dataset
faq_data = pd.read_csv(r"D:\AI_Chatbot_Project\data\cleaned_data.csv")

@app.route('/chat', methods=['POST'])
def chat():
    user_query = request.json['query'].strip().lower()

    # Handle greetings separately
    if user_query in ["hi", "hello", "hey", "hii"]:
        return jsonify({"response": "Hello 👋! I’m your college assistant. Ask me about admissions, fees, or placements."})

    # Predict intent and confidence
    query_vec = vectorizer.transform([user_query])
    probs = model.predict_proba(query_vec)[0]
    confidence = max(probs)
    predicted_intent = model.classes_[probs.argmax()]

    # Confidence threshold
    if confidence < 0.5:
        return jsonify({"response": "I’m not sure I understood that. Could you rephrase your question?"})

    # Fetch all answers for predicted intent
    answers = faq_data[faq_data['intent'] == predicted_intent]['answer'].tolist()

    if answers:
        answer = random.choice(answers)
    else:
        answer = "Sorry, I couldn’t find that information."

    return jsonify({"response": answer})

if __name__ == '__main__':
    app.run(debug=True)
