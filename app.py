import json
import os
import sqlite3
from datetime import datetime

from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load FAQs and build the TF-IDF index once at startup
with open("faq.json", encoding="utf-8") as f:
    faqs = json.load(f)

vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
faq_matrix = vectorizer.fit_transform([f["question"] for f in faqs])

# Vercel's project folder is read-only; only /tmp is writable
DB_PATH = "/tmp/logs.db" if os.environ.get("VERCEL") else "logs.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.execute("""CREATE TABLE IF NOT EXISTS logs (
                    time TEXT,
                    user_message TEXT,
                    bot_reply TEXT,
                    score REAL)""")
conn.commit()

FALLBACK = "I'm sorry, I could not find an answer to that. Could you try rephrasing your question?"
THRESHOLD = 0.2


def get_reply(message):
    vec = vectorizer.transform([message])
    scores = cosine_similarity(vec, faq_matrix)[0]
    best = int(scores.argmax())
    if scores[best] < THRESHOLD:
        return FALLBACK, 0.0
    return faqs[best]["answer"], float(scores[best])


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Message cannot be empty."}), 400

    answer, score = get_reply(message)
    conn.execute("INSERT INTO logs VALUES (?, ?, ?, ?)",
                 (datetime.utcnow().isoformat(), message, answer, score))
    conn.commit()
    return jsonify({"reply": answer, "confidence": round(score, 2)})


@app.route("/logs")
def logs():
    rows = conn.execute(
        "SELECT time, user_message, bot_reply, score FROM logs ORDER BY time DESC LIMIT 50"
    ).fetchall()
    return jsonify([
        {"time": r[0], "user_message": r[1], "bot_reply": r[2], "score": r[3]}
        for r in rows
    ])


if __name__ == "__main__":
    app.run(debug=True)
