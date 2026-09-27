# Voice-Enabled Deep Learning Chatbot — Project Report

## 1. Title
**Voice-Enabled Deep Learning Chatbot for College Assistance**

## 2. Abstract
This project implements a voice-enabled chatbot that receives spoken user queries,
transcribes them using the browser's Web Speech API, classifies the user's intent
using a Bidirectional LSTM neural network trained on a custom college-domain intent
dataset, and returns a natural-language response that is also spoken back using
speech synthesis. The system is deployed as a Flask web application. The chatbot
intelligence is a genuine deep-learning model — not a rule-based or LLM-based
system — as required by the academic brief.

## 3. Introduction
Conversational AI systems are becoming a common front-end for information services.
For a college domain (courses, admissions, exams, hostel, placements, etc.) a
lightweight intent-classification chatbot is a practical and interpretable choice.
Combining it with speech I/O in the browser gives an accessible, hands-free
experience without any native app or backend audio processing.

## 4. Problem Statement
Build a fully working web application that:
1. Accepts voice input from the microphone.
2. Converts speech to text in the browser.
3. Uses a **deep learning model** to classify the user's intent.
4. Returns a response corresponding to that intent.
5. Optionally reads the response aloud.
6. Is publicly deployable and accessible via a live URL.

## 5. Objectives
- Design an intent dataset covering ~20 college-domain intents.
- Train a neural intent classifier (Embedding + BiLSTM + Dense + Softmax).
- Serve the model behind a REST API (`POST /predict`).
- Build a modern, responsive web UI with microphone-driven input.
- Deploy the application to a free hosting platform with a public URL.

## 6. Dataset Description
- **File**: `data/intents.json`
- **Number of intents**: 19
- **Total training patterns**: ~180 sentences
- **Format**: each intent has `tag`, `patterns` (user utterances), and
  `responses` (bot replies).
- **Split**: 70 % train / 15 % validation / 15 % test, stratified by intent.

Intents covered: greeting, goodbye, thanks, about_bot, help, college, courses,
admission, timetable, attendance, exams, fees, library, hostel, placements,
contact, location, working_hours, technical_support, fallback.

## 7. Data Preprocessing
1. Lowercase every pattern.
2. Fit a Keras `Tokenizer` (vocab = 2000, OOV token `<OOV>`).
3. Convert each sentence to an integer sequence.
4. Pad/truncate every sequence to length 20 (`padding='post'`).
5. Encode intent tags with `sklearn.preprocessing.LabelEncoder`.

## 8. Speech Recognition Method
The browser's `SpeechRecognition` API (`webkitSpeechRecognition` in Chrome/Edge) is
used entirely on the client. When the user clicks the microphone button:
1. The browser requests microphone permission.
2. Audio is streamed to the browser's built-in speech engine.
3. The `onresult` event returns the recognized transcript.
4. The transcript is shown in the UI and POSTed to the backend as JSON.
No audio ever leaves the browser — this preserves privacy and removes any
server-side audio dependency.

## 9. Deep Learning Model Architecture

```
Input (integer sequence, length 20)
   │
Embedding(vocab_size, 64, mask_zero=True)
   │
Bidirectional(LSTM(64, return_sequences=True))
   │
GlobalMaxPooling1D
   │
Dense(64, activation='relu')
   │
Dropout(0.3)
   │
Dense(num_intents, activation='softmax')
```

- **Loss**: `sparse_categorical_crossentropy`
- **Optimizer**: Adam
- **Metric**: accuracy
- **Regularization**: dropout + early stopping on `val_accuracy` (patience 25)

## 10. Training Methodology
- Batch size: 16
- Max epochs: 200 (early stopping)
- Seed: 42 (for `random`, `numpy`, `tensorflow`)
- Stratified split ensures every intent appears in train/val/test.
- The best weights on validation accuracy are restored before saving.
- Artifacts saved to `models/`: `chatbot_model.keras`, `tokenizer.pkl`,
  `label_encoder.pkl`, `config.pkl`, `metrics.json`.

## 11. Chatbot Response Generation
At inference time the backend:
1. Lowercases and tokenizes the incoming message.
2. Pads it to length 20.
3. Runs a forward pass; takes `argmax` and its softmax probability.
4. If probability ≥ **0.50**, returns a random response from that intent's
   `responses` list.
5. Otherwise returns the `fallback` intent response.

## 12. System Architecture

```
┌──────────────────────────┐        ┌──────────────────────────────┐
│  Browser (HTML/CSS/JS)   │        │        Flask backend         │
│                          │        │                              │
│  Mic → Web Speech API    │        │  /predict                    │
│           │              │        │    ├─ tokenize + pad         │
│           ▼              │  JSON  │    ├─ BiLSTM.predict()       │
│  Recognized text ────────┼───────▶│    ├─ threshold check        │
│                          │        │    └─ response from intents  │
│  Bot bubble ◀────────────┼────────┤                              │
│  SpeechSynthesis         │        │  /health, /api/info          │
└──────────────────────────┘        └──────────────────────────────┘
```

## 13. Implementation
- **Backend**: `app.py` loads the trained model, tokenizer and label encoder
  **once** at startup — never per request. Deployed with Gunicorn.
- **Training**: `train.py` is a standalone script; it is re-run during
  deployment (see `render.yaml`).
- **Frontend**: single-page `templates/index.html` with `static/style.css`
  and `static/script.js`. The UI shows the recognized speech, predicted
  intent, and confidence in a technical panel to prove that the neural
  network is actually being used.

## 14. Results
> Do not fabricate metrics. Fill this section from `models/metrics.json`
> after running `python train.py`.

| Split      | Loss | Accuracy |
|------------|------|----------|
| Train      | _fill_ | _fill_ |
| Validation | _fill_ | _fill_ |
| Test       | _fill_ | _fill_ |

Example predictions after training:

| Input | Predicted intent | Response |
|-------|------------------|----------|
| "Hello" | greeting | "Hello! How can I help you today?" |
| "What courses do you offer?" | courses | "We offer undergraduate programs..." |
| "How can I check my attendance?" | attendance | "Your attendance is available on..." |
| "When are the exams?" | exams | "Mid-semester exams are held..." |
| "Tell me about placements." | placements | "Our placement cell has..." |
| "Thank you" | thanks | "You're welcome!" |

## 15. Screenshots
Add screenshots of the running UI here (chat window, listening state,
technical panel showing intent and confidence).

## 16. Deployment
Deployment is preconfigured for Render (`render.yaml` + `Procfile`) and
also supported on Hugging Face Spaces (Docker) and Railway. After deployment,
paste the resulting public URL here.

**Live URL:** _to be filled by the deployer_

## 17. Limitations
- The dataset is small (~180 patterns); vocabulary outside the training
  distribution may be routed to the fallback intent.
- Web Speech API is only fully supported in Chromium-based browsers.
- Free-tier hosting can cold-start after inactivity.
- No conversational memory — each request is stateless.

## 18. Future Scope
- Expand the dataset and add multilingual intents.
- Replace the softmax classifier with a transformer encoder (e.g., DistilBERT).
- Add dialogue state tracking for multi-turn conversations.
- Server-side ASR (Whisper) for browsers without Web Speech API.
- Integrate with the college's real student portal for personalized answers
  (attendance, timetable, fees).

## 19. Conclusion
The project delivers a complete voice-in / voice-out chatbot that uses a
genuine deep-learning intent classifier. It satisfies all the academic
requirements: real neural network, real dataset, real training pipeline,
real REST API, real UI, and a repeatable deployment path.

## 20. References
- Chollet, F. *Deep Learning with Python* (2nd ed.), Manning, 2021.
- Hochreiter, S. & Schmidhuber, J. "Long Short-Term Memory." *Neural Computation*, 1997.
- TensorFlow / Keras documentation: <https://www.tensorflow.org/>
- MDN Web Docs — Web Speech API: <https://developer.mozilla.org/docs/Web/API/Web_Speech_API>
- Flask documentation: <https://flask.palletsprojects.com/>
