# Voice-Enabled Deep Learning Chatbot
### A Complete Project Documentation

**Author:** Kushagra
**Date:** September 2026
**Course:** [Your course / lab code]
**Live URL:** https://voice-chatbot-nlnq.onrender.com

---

## Table of Contents

1. Abstract
2. Objectives
3. Technologies Used
4. System Architecture
5. Dataset
6. Data Preprocessing
7. Deep Learning Model Architecture
8. Training Methodology
9. Speech Recognition
10. Text-to-Speech
11. Backend REST API
12. Frontend Interface
13. Project Structure
14. Complete Setup Commands
15. Deployment
16. Demonstration Guide
17. Results
18. Limitations
19. Future Enhancements
20. Conclusion
21. References

---

## 1. Abstract

This project implements a fully working **voice-enabled chatbot** that listens to the user through the browser microphone, converts the speech into text using the Web Speech API, classifies the user's intent using a **Bidirectional LSTM neural network** trained with TensorFlow/Keras, generates a natural-language response, and speaks the response back using the browser's SpeechSynthesis API. The complete system is deployed to a public URL on **Render**. The chatbot's intelligence is a genuine deep-learning intent classifier — not keyword matching, not rule-based logic, and not an external LLM API — as required by the academic brief.

---

## 2. Objectives

- Design and hand-author a custom college-domain intent dataset.
- Train a neural intent classifier (Embedding + BiLSTM + Dense + Softmax).
- Build a REST API around the trained model using Flask.
- Build a modern, responsive web UI supporting both microphone and text input.
- Deploy the entire application to a public URL free of cost.
- Demonstrate the end-to-end pipeline Voice → Text → Neural Network → Response → Speech.

---

## 3. Technologies Used

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Deep Learning | TensorFlow / Keras |
| Numerical | NumPy |
| Preprocessing | scikit-learn (LabelEncoder, train_test_split) |
| Web framework | Flask |
| WSGI server | Gunicorn |
| Frontend | HTML5, CSS3, vanilla JavaScript |
| Speech input | Web Speech API (`SpeechRecognition` / `webkitSpeechRecognition`) |
| Speech output | Web Speech API (`SpeechSynthesis`) |
| Version control | Git + GitHub |
| Hosting | Render (Blueprint deployment via `render.yaml`) |
| Deployment configs | `Procfile`, `runtime.txt`, `render.yaml` |

---

## 4. System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                       BROWSER (client)                         │
│                                                                │
│   ┌────────────────┐   Web Speech API   ┌─────────────────┐    │
│   │  Microphone    │ ─────────────────▶ │  Recognized     │    │
│   │  (user voice)  │                    │  text string    │    │
│   └────────────────┘                    └────────┬────────┘    │
│                                                  │             │
│   ┌────────────────┐                             │             │
│   │  Text input    │ ────────────────────────────┤             │
│   └────────────────┘                             ▼             │
│                                          fetch POST /predict   │
│                                          {message: "..."}      │
└─────────────────────────────────────────────────┬──────────────┘
                                                  │
                                                  ▼
┌───────────────────────────────────────────────────────────────┐
│                    RENDER (server)                            │
│                                                               │
│   Flask (app.py) — Gunicorn — Python 3.11                     │
│                                                               │
│   POST /predict                                               │
│     ├─ lowercase + strip                                      │
│     ├─ Keras Tokenizer → integer sequence                     │
│     ├─ pad_sequences(maxlen=20)                               │
│     ├─ BiLSTM.__call__(x)  ────▶ softmax probabilities        │
│     ├─ argmax + confidence check (threshold = 0.50)           │
│     └─ return {intent, confidence, response}                  │
│                                                               │
│   GET  /health      → {"status":"ok"}                         │
│   GET  /api/info    → model + metrics metadata                │
│   GET  /            → serves templates/index.html             │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
                     JSON response to browser
                                │
                                ▼
        Chat bubble rendered + SpeechSynthesis reads it aloud
```

**Key design decision:** the trained model, tokenizer, and label encoder are loaded **once at process startup** and warmed with a dummy inference. This keeps every user request in the 500ms – 2s range on the free tier — no per-request model reload.

---

## 5. Dataset

- **File:** `data/intents.json`
- **Number of intents:** 19
- **Number of training patterns:** ~180 hand-written sentences
- **Format:** each intent has three fields — `tag`, `patterns` (user utterances), and `responses` (bot replies).
- **Split:** 70% train / 15% validation / 15% test, **stratified** by intent so every class appears in every split.

### Intents covered

| Category | Intents |
|---|---|
| Conversational | greeting, goodbye, thanks, about_bot, help |
| Academic | courses, admission, timetable, attendance, exams |
| Facilities | fees, library, hostel |
| Institutional | college, placements, contact, location, working_hours |
| Safety net | technical_support, fallback |

### Sample intent entry

```json
{
  "tag": "courses",
  "patterns": [
    "What courses do you offer",
    "List the courses",
    "Which programs are available",
    "Undergraduate courses",
    "Postgraduate courses",
    "..."
  ],
  "responses": [
    "We offer undergraduate programs (B.Tech, B.Sc, BBA, BCA) and postgraduate programs (M.Tech, M.Sc, MBA, MCA)..."
  ]
}
```

---

## 6. Data Preprocessing

The training and inference pipelines apply **identical** preprocessing (train/serve parity):

1. **Lowercase** every sentence.
2. **Whitespace strip.**
3. **Tokenization** using Keras `Tokenizer` (vocabulary size = 2000, out-of-vocabulary token `<OOV>`).
4. **Integer sequence conversion.**
5. **Padding** to a fixed length of 20 tokens using `pad_sequences(padding='post', truncating='post')`.
6. **Label encoding** of intent tags using `sklearn.preprocessing.LabelEncoder`.

Any user query passed to `/predict` follows the exact same steps before hitting the model.

---

## 7. Deep Learning Model Architecture

```
Input (integer sequence, length 20)
        │
Embedding(vocab_size=2000, output_dim=64, mask_zero=True)
        │
Bidirectional( LSTM(64, return_sequences=True) )
        │
GlobalMaxPooling1D
        │
Dense(64, activation='relu')
        │
Dropout(0.3)
        │
Dense(num_intents=19, activation='softmax')
        │
Predicted intent + confidence
```

**Hyperparameters**

| Parameter | Value |
|---|---|
| Vocabulary size | 2000 |
| Embedding dimension | 64 |
| Sequence length (max_len) | 20 |
| LSTM units | 64 (bidirectional → 128 effective) |
| Dropout | 0.3 |
| Optimizer | Adam |
| Loss | Sparse Categorical Cross-Entropy |
| Metric | Accuracy |
| Batch size | 16 |
| Max epochs | 200 (with early stopping) |
| Early-stopping monitor | `val_accuracy`, patience = 25, restore best weights |
| Random seed | 42 (Python, NumPy, TensorFlow) |
| Confidence threshold | 0.50 |

If confidence < 0.50, the backend returns the `fallback` intent response instead of guessing.

---

## 8. Training Methodology

- Executed by `train.py`.
- Loads `data/intents.json`, builds tokenizer and label encoder.
- Stratified 70/15/15 train/val/test split.
- Trains with early stopping on validation accuracy.
- After training, saves five artifacts to `models/`:
  - `chatbot_model.keras`
  - `tokenizer.pkl`
  - `label_encoder.pkl`
  - `config.pkl` — max_len, vocab_size, confidence_threshold
  - `metrics.json` — final train/val/test loss and accuracy
- **Training runs automatically during cloud deployment** (see `render.yaml` build command). This guarantees the deployed model exactly matches the checked-in dataset.

---

## 9. Speech Recognition

Uses the **Web Speech API** (`SpeechRecognition` — vendor-prefixed `webkitSpeechRecognition` in Chromium browsers). The recognition is done **entirely in the browser** by the browser vendor's speech engine — no audio ever leaves the client.

**Sequence when the user clicks the mic:**

1. Browser requests microphone permission (once).
2. `recognition.start()` begins listening.
3. UI shows the "Listening…" indicator (red pulsing dot).
4. `onresult` event fires with the transcript.
5. Transcript is displayed and posted to `/predict`.
6. `onend` event resets the UI.

**Advantages of client-side ASR:**
- Zero backend audio processing cost.
- Privacy: raw audio never leaves the browser.
- Works cross-platform (Windows, macOS, Android) with the same code.

---

## 10. Text-to-Speech

After every bot response the frontend calls `window.speechSynthesis.speak(new SpeechSynthesisUtterance(response))`. A speaker button (🔊) also replays the last response on demand. This completes the voice-in / voice-out loop.

---

## 11. Backend REST API

Implemented in `app.py`. The model, tokenizer, and label encoder are loaded once at startup and a warm-up inference is executed so first-user latency is low.

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Serves the chatbot UI (`templates/index.html`) |
| `/health` | GET | Liveness check — returns `{"status":"ok"}` |
| `/api/info` | GET | Returns model metadata + `metrics.json` |
| `/predict` | POST | Core inference endpoint |

### `/predict` example

Request:
```json
POST /predict
Content-Type: application/json
{ "message": "What courses do you offer?" }
```

Response:
```json
{
  "intent": "courses",
  "confidence": 0.9421,
  "response": "We offer undergraduate programs...",
  "below_threshold": false,
  "threshold": 0.5
}
```

If the confidence is below the threshold:
```json
{
  "intent": "fallback",
  "confidence": 0.3182,
  "response": "I'm not completely sure I understood that. Could you please rephrase your question?",
  "below_threshold": true,
  "threshold": 0.5
}
```

---

## 12. Frontend Interface

Built with plain HTML, CSS, and JavaScript (no framework — small footprint, fast load).

**Features**

- Modern dark UI with gradient accents and glassmorphism card.
- Large microphone button that pulses red when listening.
- Text input as a fallback for evaluators.
- Chat bubbles with fade-in animation.
- **Technical panel** showing predicted intent + confidence — proves the deep-learning model is actively being used.
- Status pill showing Ready / Listening / Thinking states.
- Responsive breakpoints for desktop, laptop, and mobile.

---

## 13. Project Structure

```
voice-chatbot/
├── app.py                       Flask backend (loads model at startup)
├── train.py                     Trains BiLSTM and saves artifacts
├── requirements.txt             Python dependencies
├── Procfile                     Gunicorn startup command
├── render.yaml                  Render Blueprint (build + start commands)
├── runtime.txt                  Python 3.11.9 pin
├── .gitignore
├── README.md
├── data/
│   └── intents.json             19 intents, ~180 patterns
├── models/                      (generated by train.py)
│   ├── chatbot_model.keras
│   ├── tokenizer.pkl
│   ├── label_encoder.pkl
│   ├── config.pkl
│   └── metrics.json
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── script.js
└── report/
    └── project_report.md
```

---

## 14. Complete Setup Commands

### Run locally (Python 3.11 recommended)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py
python app.py
# open http://localhost:5000 in Chrome or Edge
```

### Push to GitHub (one-time setup)

```powershell
cd "c:\Users\2222619\OneDrive - MyFedEx\Desktop\UI"
git init
git branch -M main
git add .
git commit -m "Initial commit: voice-enabled deep learning chatbot"
git remote add origin https://github.com/<your-username>/voice-chatbot.git
git push -u origin main
```

### Push code updates after initial deploy

```powershell
git add .
git commit -m "Describe the change"
git push
```

Render auto-detects the push and redeploys.

---

## 15. Deployment

### Configuration files

- **`render.yaml`** — declares the service type (web), Python env, build command (`pip install -r requirements.txt && python train.py`), and start command (Gunicorn).
- **`Procfile`** — `web: gunicorn app:app --workers 1 --threads 1 --timeout 300 --graceful-timeout 60 --bind 0.0.0.0:$PORT`
- **`runtime.txt`** — pins the deployment Python to 3.11.9.

### Deployment steps

1. Push the repository to a **public** GitHub repo.
2. Sign in at https://render.com with GitHub.
3. Click **New +** → **Blueprint** → select the repo.
4. Render installs dependencies, runs `python train.py`, and starts Gunicorn (~5–7 minutes for the first build).
5. When the badge turns green (Live), the public URL is displayed at the top of the service page.

### Free-tier characteristics

- **Cold start:** first request after ~15 minutes of inactivity takes 60–120 seconds (TensorFlow load + graph tracing).
- **Warm requests:** 500 ms – 2 s per `/predict`.
- **Warm-up trick:** open `/health` in a browser tab a minute before the demo to wake the server.

### Live URL

> https://voice-chatbot-nlnq.onrender.com

---

## 16. Demonstration Guide

### Before the demo (2 minutes before)

1. Open a fresh tab → go to `https://voice-chatbot-nlnq.onrender.com/health`.
2. Wait for `{"status":"ok"}`.
3. Open a second tab with the homepage. You are ready.

### Voice queries to demo (all trained; all should hit ≥ 70% confidence)

| # | Say into the mic | Predicted intent |
|---|---|---|
| 1 | "Hello" | greeting |
| 2 | "Who are you and what can you do?" | about_bot |
| 3 | "What courses do you offer?" | courses |
| 4 | "How do I get admission?" | admission |
| 5 | "When are the exams?" | exams |
| 6 | "How can I check my attendance?" | attendance |
| 7 | "What is the fee structure?" | fees |
| 8 | "Tell me about placements" | placements |
| 9 | "Where is the library?" | library |
| 10 | "Tell me about the hostel" | hostel |
| 11 | "How can I contact the college?" | contact |
| 12 | "Where is the college located?" | location |
| 13 | "The portal is not working" | technical_support |
| 14 | "Thank you so much" | thanks |
| 15 | "Goodbye" | goodbye |

### Show the fallback path (proves confidence threshold works)

Ask something unrelated:
- "Can you order me a pizza?"
- "What is the meaning of life?"

The bot should respond with the fallback message and the technical panel should show low confidence.

### Talking points while the bot is responding

- *"The technical panel below the chat shows the predicted intent and confidence from the softmax layer — that's proof a real neural network is doing the classification."*
- *"Speech recognition runs in the browser via the Web Speech API — no audio ever leaves the client."*
- *"The recognized text is POSTed to a Flask endpoint at `/predict`."*
- *"The backend runs a Bidirectional LSTM: Embedding → BiLSTM → GlobalMaxPooling → Dense → Softmax over 19 intents."*
- *"If the softmax confidence is below 50%, the system returns a fallback response instead of guessing."*
- *"The response is spoken back using SpeechSynthesis — full voice-in, voice-out loop."*
- *"This is a live public URL deployed on Render, not a localhost demo."*

### Bonus — show the API is live

Open `https://voice-chatbot-nlnq.onrender.com/api/info` in a new tab. This returns JSON with the model configuration, class list, and training metrics. Point out that this proves the model is loaded server-side.

---

## 17. Results

> Fill this table from `models/metrics.json` after training. Do **not** fabricate numbers.

| Split | Loss | Accuracy |
|---|---|---|
| Train | _fill_ | _fill_ |
| Validation | _fill_ | _fill_ |
| Test | _fill_ | _fill_ |

Read the exact values from the live app at:
```
https://voice-chatbot-nlnq.onrender.com/api/info
```

### Example predictions after training

| Input | Predicted intent | Response |
|---|---|---|
| "Hello" | greeting | "Hello! How can I help you today?" |
| "What courses do you offer?" | courses | "We offer undergraduate programs..." |
| "How can I check my attendance?" | attendance | "Your attendance is available on..." |
| "When are the exams?" | exams | "Mid-semester exams are held..." |
| "Tell me about placements." | placements | "Our placement cell has..." |
| "Thank you" | thanks | "You're welcome!" |

---

## 18. Limitations

- **Small dataset:** ~180 patterns per 19 intents means unusual phrasings can be misclassified.
- **English-only:** the tokenizer and dataset are both English.
- **Browser support for ASR:** Web Speech API is supported reliably only in Chromium browsers (Chrome, Edge). Firefox and iOS Safari are not fully supported.
- **Free-tier cold start:** Render's free plan sleeps the container after 15 minutes of inactivity; first request after that takes 60–120 s.
- **Stateless:** the bot has no conversational memory — each request is independent.
- **No spelling correction:** typos may be routed to the fallback intent.

---

## 19. Future Enhancements

### Short-term (dataset + preprocessing)
- Expand the dataset to 500+ patterns (3× current).
- Add lemmatization (NLTK WordNet) so "running" and "run" collapse to one token.
- Add synonym-based data augmentation to triple the effective dataset.
- Add typo tolerance using Levenshtein-based fuzzy matching on the OOV path.

### Medium-term (model)
- Replace random Embedding with pretrained **GloVe** or **FastText** vectors — captures semantic similarity between unseen words.
- Try a **Transformer encoder** (DistilBERT) via Hugging Face for better generalization on paraphrased queries.
- Add character-level embeddings alongside word embeddings for robustness to misspellings.

### Long-term (product)
- **Multi-turn dialogue** with conversational state tracking.
- **Multilingual support** (Hindi, Tamil, etc.) with a language-detection front-stage.
- **Personalisation:** connect to the college portal so the bot can answer "What is *my* attendance?" for the logged-in student.
- **Server-side ASR** using OpenAI Whisper for browsers without Web Speech API and for higher accuracy in noisy environments.
- **Analytics dashboard** — track which intents get asked most, which get low confidence, and use that data to improve the training set continuously.
- **Voice cloning / custom TTS** using a lightweight open model like Piper for a college-branded voice.

### Deployment enhancements
- Move to Hugging Face Spaces for better free-tier CPU allocation.
- Add UptimeRobot pings so the demo URL never cold-starts.
- Containerise with Docker for reproducible deploys on any platform.

---

## 20. Conclusion

The project delivers a complete voice-in / voice-out chatbot backed by a genuine deep-learning intent classifier. Every requirement of the brief is met:

- Speech recognition ✔ (Web Speech API)
- Deep learning intent classification ✔ (BiLSTM in TensorFlow/Keras)
- Response generation from a dataset ✔ (intents.json)
- REST API ✔ (Flask + Gunicorn)
- Modern responsive UI ✔ (HTML/CSS/JS)
- Public live URL ✔ (Render deployment)
- No LLM API, no rule-based logic, no keyword matching ✔

The system is production-shaped: reproducible training, versioned artifacts, cloud deployment configuration, health checks, and an explicit confidence threshold with graceful fallback.

---

## 21. References

- Chollet, F. *Deep Learning with Python* (2nd ed.), Manning, 2021.
- Hochreiter, S. & Schmidhuber, J. "Long Short-Term Memory." *Neural Computation*, 1997.
- Vaswani, A. et al. "Attention Is All You Need." *NeurIPS*, 2017.
- TensorFlow documentation — https://www.tensorflow.org/
- Keras documentation — https://keras.io/
- Flask documentation — https://flask.palletsprojects.com/
- MDN — Web Speech API — https://developer.mozilla.org/docs/Web/API/Web_Speech_API
- MDN — SpeechRecognition — https://developer.mozilla.org/docs/Web/API/SpeechRecognition
- MDN — SpeechSynthesis — https://developer.mozilla.org/docs/Web/API/SpeechSynthesis
- Render documentation — https://render.com/docs
- Scikit-learn documentation — https://scikit-learn.org/

---

*End of document.*
