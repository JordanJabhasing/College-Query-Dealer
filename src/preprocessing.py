import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (only needed once)
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Remove non-letters
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    # Lowercase + split
    tokens = text.lower().split()
    # Lemmatize and remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# 👉 IMPORTANT: Update this path to where your dataset is saved
file_path = r"D:\AI_Chatbot_Project\data\college_faq_dataset.csv"

# Load dataset
df = pd.read_csv(file_path)

# Clean the questions
df['cleaned_question'] = df['question'].apply(clean_text)

# Save cleaned dataset
output_path = r"D:\AI_Chatbot_Project\data\cleaned_data.csv"
df.to_csv(output_path, index=False)

print(f"✅ Cleaned data saved successfully to {output_path}")
