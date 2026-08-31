# College-Query-Dealer
This is a Minor Project i did for my Internship, It is a AI Chatbot that can be used in college web portals to easily answer the student queries, instead of stressing the teachers out. All the files are tested but with minimals. This is my first time creating an actual Software project. So Kindly Let me know if theres any issue with the project.
This Chatbot was not easy to create, as this is my first time doing a project, It was really a big step up for me.
I tried Youtube videos, then AI chatbot websites, was not satisfied, then with the help of "copilot" i did this big job. I created each and every file, longed for perfection, gathered it's knowledge from actual college students through g-forms, and trained it and got it to work. i know this is not much, but foe me it is.
I learned many things while working on this project.

🧑‍🎓 College Chatbot

📌 Overview
This project is a College Assistant Chatbot designed to answer student queries about admissions, courses, fees, internships, and placements. It integrates Natural Language Processing (NLP) techniques with a web-based interface to provide a smooth, conversational experience.

The chatbot combines:
Intent classification using a trained ML model.
Fuzzy matching to handle spelling mistakes and near matches.
Semantic search with embeddings for meaning-based query resolution.
Keyword-based intent guard for reliable routing.
A Flask backend serving responses via REST API.
A frontend (HTML/CSS/JS) with light/dark mode toggle and chat UI.

Features
Handles spelling mistakes (e.g., “admiisions” → “admissions”).
Multi-layer fallback system:
Keyword intent check.
ML intent classifier (with confidence threshold).
Fuzzy matching.
Semantic search.
Always returns a valid response (no crashes).
Modern UI with:
Campus-themed background.
Light/Dark mode toggle.
Chat bubbles for user and bot.
Extensible dataset – FAQ data stored in cleaned_data.csv.

Architecture
- Frontend (index.html + CSS + JS)
Chat interface.
Dark mode toggle.
Sends queries to Flask backend via fetch().
- Backend (Flask app.py)
Loads trained intent model + vectorizer.
Loads FAQ dataset.
Embedding model (SentenceTransformer).
Routes queries through keyword → ML → fuzzy → semantic pipeline.
- Data (cleaned_data.csv)
Contains intents, questions, and answers.
Used for training and response lookup.
- Models
intent_model.pkl – trained classifier.
vectorizer.pkl – text vectorizer.

System Architecture

User Query
    │
    ▼
Frontend (HTML/CSS/JS)
    │   - Chat UI
    │   - Dark/Light mode toggle
    │   - Sends query via fetch()
    ▼
Flask Backend (app.py)
    │
    ├── Keyword Guard
    │       (direct intent detection)
    │
    ├── ML Intent Classifier
    │       (confidence ≥ 0.5 → answer)
    │
    ├── Fuzzy Matching
    │       (handles typos, near matches)
    │
    └── Semantic Search
            (embeddings for meaning-based answers)
    ▼
FAQ Dataset (cleaned_data.csv)
    │   - Intents
    │   - Questions
    │   - Answers
    ▼
Response Returned
    │
    ▼
Frontend Chat UI
    │   - User bubble (blue)
    │   - Bot bubble (orange/green in dark mode)

How to Run
Clone the project folder.
Install dependencies:
pip install flask flask-cors pandas rapidfuzz sentence-transformers
Start the backend:
python src/app.py
Open index.html in a browser.
Type queries and interact with the chatbot.

Research & Contributions
Explored NLP intent classification using scikit-learn.
Integrated fuzzy matching (RapidFuzz) for typo handling.
Added semantic embeddings (MiniLM) for meaning-based search.
Designed keyword guard to prevent intent confusion.
Built frontend UI with dark mode and responsive design.
Debugged Flask errors (ensured all routes return JSON). 

Future Improvements
Expand dataset with more diverse queries.
Add contextual memory (multi-turn conversations).
Deploy on cloud (Heroku/AWS) for public access.
Integrate with college database for live data.
Add voice input/output for accessibility.

👨‍💻 Author

Jordan – Technical Advisor & AI Intern
Minor Project – AI Chatbot for Student Queries – NLP intent classification demo


