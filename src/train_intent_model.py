import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset (make sure it has intent, question, answer, cleaned_question)
df = pd.read_csv(r"D:\AI_Chatbot_Project\data\cleaned_data.csv")

X = df['cleaned_question']
y = df['intent']

# Vectorize
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_vec, y)

# Save model + vectorizer together
pickle.dump(model, open(r"D:\AI_Chatbot_Project\models\intent_model.pkl", 'wb'))
pickle.dump(vectorizer, open(r"D:\AI_Chatbot_Project\models\vectorizer.pkl", 'wb'))

print("✅ Model and vectorizer saved successfully!")