import json
import os
import re
import sqlite3
import uuid
from collections import defaultdict, deque
from datetime import datetime

from flask import Flask, request, jsonify, render_template
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# NLTK PorterStemmer needs no downloaded data, so it works on Vercel.
stemmer = PorterStemmer()

# Small stopword list. Question words (how, what, can) are kept on purpose,
# because they carry meaning in FAQ questions.
STOPWORDS = {"a", "an", "the", "is", "are", "am", "do", "does", "i", "my", "me",
             "to", "of", "on", "in", "at", "for", "please", "it", "be", "your", "our"}


def analyze(text):
    """Lowercase, tokenize, remove stopwords, stem, and add bigrams."""
    tokens = [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]
    stems = [stemmer.stem(t) for t in tokens]
    bigrams = [f"{a}_{b}" for a, b in zip(stems, stems[1:])]
    return stems + bigrams


with open("faq.json", encoding="utf-8") as f:
    faqs = json.load(f)

vectorizer = TfidfVectorizer(analyzer=analyze)
faq_matrix = vectorizer.fit_transform([f["question"] for f in faqs])

# Vercel's project folder is read-only; only /tmp is writable
DB_PATH = "/tmp/chat_logs.db" if os.environ.get("VERCEL") else "chat_logs.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.execute("""CREATE TABLE IF NOT EXISTS chat_logs (
                    time TEXT,
                    session_id TEXT,
                    user_message TEXT,
                    bot_reply TEXT,
                    matched_question TEXT,
                    score REAL)""")
conn.commit()

FALLBACK = "I'm sorry, I could not find an answer to that. Could you try rephrasing your question?"
THRESHOLD = 0.2
HISTORY_SIZE = 6

# session_id -> recent (user_message, faq_index or None) pairs
sessions = defaultdict(lambda: deque(maxlen=HISTORY_SIZE))


def match(text):
    scores = cosine_similarity(vectorizer.transform([text]), faq_matrix)[0]
    idx = int(scores.argmax())
    return idx, float(scores[idx])


def respond(message, history):
    """Return (faq_index, score), or (None, 0.0) if nothing matches."""
    idx, score = match(message)
    if score >= THRESHOLD:
        return idx, score

    # Follow-up support: retry using the previous message as context,
    # so "and what about refunds?" can be understood after a related question.
    if history:
        previous_message, _ = history[-1]
        idx, score = match(previous_message + " " + message)
        if score >= THRESHOLD:
            return idx, score

    return None, 0.0


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()[:500]
    if not message:
        return jsonify({"error": "Message cannot be empty."}), 400

    session_id = data.get("session_id") or str(uuid.uuid4())
    history = sessions[session_id]

    idx, score = respond(message, history)
    if idx is None:
        reply, matched, score = FALLBACK, "", 0.0
    else:
        reply = faqs[idx]["answer"]
        matched = faqs[idx]["question"]

    history.append((message, idx))

    conn.execute("INSERT INTO chat_logs VALUES (?, ?, ?, ?, ?, ?)",
                 (datetime.utcnow().isoformat(), session_id, message, reply, matched, score))
    conn.commit()

    return jsonify({"reply": reply, "confidence": round(score, 2), "session_id": session_id})


@app.route("/logs")
def logs():
    rows = conn.execute("""SELECT time, session_id, user_message, bot_reply, matched_question, score
                           FROM chat_logs ORDER BY time DESC LIMIT 50""").fetchall()
    keys = ["time", "session_id", "user_message", "bot_reply", "matched_question", "score"]
    return jsonify([dict(zip(keys, r)) for r in rows])


if __name__ == "__main__":
    app.run(debug=True)
