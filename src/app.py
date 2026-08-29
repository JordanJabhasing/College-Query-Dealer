import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import random
from rapidfuzz import process
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
CORS(app)

# Load intent model + vectorizer (baseline classifier)
model = pickle.load(open(r"D:\AI_Chatbot_Project\models\intent_model.pkl", 'rb'))
vectorizer = pickle.load(open(r"D:\AI_Chatbot_Project\models\vectorizer.pkl", 'rb'))

# Load dataset
faq_data = pd.read_csv(r"D:\AI_Chatbot_Project\data\cleaned_data.csv")

# Load semantic embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')
question_embeddings = embedder.encode(faq_data['question'].tolist(), convert_to_tensor=True)

def fuzzy_match(user_query, questions):
    best_match, score, _ = process.extractOne(user_query, questions)
    if score > 70:  # threshold for fuzzy match
        idx = questions.index(best_match)
        return faq_data.iloc[idx]['answer']
    return None

def semantic_search(user_query):
    query_embedding = embedder.encode(user_query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, question_embeddings)[0]
    best_idx = scores.argmax().item()
    return faq_data.iloc[best_idx]['answer']

@app.route('/chat', methods=['POST'])
def chat():
    user_query = request.json['query'].strip().lower()

    # Handle greetings separately
    if user_query in ["hi", "hello", "hey", "hii"]:
        return jsonify({"response": "Hello 👋! I’m your college assistant. Ask me about admissions, fees, or placements."})

    # Predict intent + confidence
    query_vec = vectorizer.transform([user_query])
    probs = model.predict_proba(query_vec)[0]
    confidence = max(probs)
    predicted_intent = model.classes_[probs.argmax()]

    # If confidence is high → use intent classifier
    if confidence >= 0.4:
        answers = faq_data[faq_data['intent'] == predicted_intent]['answer'].tolist()
        if answers:
            return jsonify({"response": random.choice(answers)})

    # If confidence is low → try fuzzy match
    fuzzy_answer = fuzzy_match(user_query, faq_data['question'].tolist())
    if fuzzy_answer:
        return jsonify({"response": fuzzy_answer})

    # If fuzzy fails → fall back to semantic search
    semantic_answer = semantic_search(user_query)
    return jsonify({"response": semantic_answer})

if __name__ == '__main__':
    app.run(debug=True)
