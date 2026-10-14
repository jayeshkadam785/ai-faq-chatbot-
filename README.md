# AI-Powered FAQ Chatbot

A customer support chatbot that answers common questions using TF-IDF text matching and cosine similarity. Each conversation is logged to a SQLite database.

## Technologies
- Python, Flask
- scikit-learn (TF-IDF vectorizer, cosine similarity)
- SQLite (interaction logs)
- HTML, CSS, JavaScript (chat interface)

## How it works
1. The FAQ questions are converted into TF-IDF vectors when the server starts.
2. A user's message is converted into the same vector space.
3. Cosine similarity finds the closest FAQ question.
4. If the match is strong enough, the stored answer is returned. Otherwise, a fallback message is shown.
5. Every message and reply is saved to the `logs` table.

## Setup
pip install -r requirements.txt
python app.py

Open http://127.0.0.1:5000 in your browser.

## Endpoints
- `GET /` returns the chat interface
- `POST /chat` takes `{"message": "..."}` and returns `{"reply": "...", "confidence": 0.0}`
- `GET /logs` returns the 50 most recent interactions

## Limitations and next steps
- Matching is based on keywords, so very different wordings may not match. A transformer model could improve this.
- On Vercel, logs are stored in `/tmp`, which is temporary. A hosted database such as PostgreSQL would keep logs permanently.
