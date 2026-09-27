"""
Flask backend for the Voice-Enabled Deep Learning Chatbot.

Loads the trained BiLSTM intent classifier once at startup and exposes:
    GET  /              -> chatbot UI
    GET  /health        -> health check
    GET  /api/info      -> model metadata
    POST /predict       -> intent classification + response
"""

from __future__ import annotations

import os

# Keep TensorFlow's memory footprint under the 512 MB Render free-tier limit.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")

import json
import pickle
import random
from typing import Any

import numpy as np
from flask import Flask, jsonify, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "intents.json")

app = Flask(__name__, static_folder="static", template_folder="templates")

# ---------------------------------------------------------------------------
# Load model + artifacts once at startup
# ---------------------------------------------------------------------------

def _load_artifacts() -> dict[str, Any]:
    model_path = os.path.join(MODELS_DIR, "chatbot_model.keras")
    tokenizer_path = os.path.join(MODELS_DIR, "tokenizer.pkl")
    label_path = os.path.join(MODELS_DIR, "label_encoder.pkl")
    config_path = os.path.join(MODELS_DIR, "config.pkl")

    for p in (model_path, tokenizer_path, label_path, config_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Required artifact missing: {p}. Run `python train.py` first."
            )

    model = load_model(model_path)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    with open(label_path, "rb") as f:
        label_encoder = pickle.load(f)
    with open(config_path, "rb") as f:
        config = pickle.load(f)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        intents = json.load(f)["intents"]
    responses_by_tag = {i["tag"]: i["responses"] for i in intents}

    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "label_encoder": label_encoder,
        "config": config,
        "responses_by_tag": responses_by_tag,
        "metrics": metrics,
    }


ARTIFACTS = _load_artifacts()


# Warm up the model so the first real /predict call is fast (avoids ~15s TF tracing lag).
try:
    _warm_seq = pad_sequences(
        ARTIFACTS["tokenizer"].texts_to_sequences(["hello"]),
        maxlen=int(ARTIFACTS["config"]["max_len"]),
        padding="post",
        truncating="post",
    )
    ARTIFACTS["model"](_warm_seq, training=False)
    print("[app] Model warmup complete.")
except Exception as _exc:  # noqa: BLE001
    print(f"[app] Model warmup skipped: {_exc}")


def predict_intent(message: str) -> dict[str, Any]:
    text = (message or "").lower().strip()
    tokenizer = ARTIFACTS["tokenizer"]
    model = ARTIFACTS["model"]
    label_encoder = ARTIFACTS["label_encoder"]
    config = ARTIFACTS["config"]
    responses_by_tag = ARTIFACTS["responses_by_tag"]

    max_len = int(config["max_len"])
    threshold = float(config.get("confidence_threshold", 0.5))

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    # Direct __call__ avoids the Keras predict() per-batch overhead — noticeably faster on tiny inputs.
    probs = np.asarray(model(padded, training=False))[0]

    idx = int(np.argmax(probs))
    confidence = float(probs[idx])
    intent_tag = str(label_encoder.classes_[idx])

    if confidence < threshold:
        fallback = responses_by_tag.get(
            "fallback",
            ["I'm not completely sure I understood that. Could you please rephrase your question?"],
        )
        return {
            "intent": "fallback",
            "confidence": round(confidence, 4),
            "response": random.choice(fallback),
            "below_threshold": True,
            "threshold": threshold,
        }

    responses = responses_by_tag.get(intent_tag, ["Sorry, I don't have an answer for that."])
    return {
        "intent": intent_tag,
        "confidence": round(confidence, 4),
        "response": random.choice(responses),
        "below_threshold": False,
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/info", methods=["GET"])
def api_info():
    metrics = ARTIFACTS["metrics"]
    return jsonify(
        {
            "model": "BiLSTM intent classifier",
            "classes": list(ARTIFACTS["label_encoder"].classes_),
            "num_classes": int(len(ARTIFACTS["label_encoder"].classes_)),
            "max_len": int(ARTIFACTS["config"]["max_len"]),
            "confidence_threshold": float(ARTIFACTS["config"].get("confidence_threshold", 0.5)),
            "metrics": metrics,
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(silent=True) or {}
        message = payload.get("message", "")
        if not isinstance(message, str) or not message.strip():
            return jsonify({"error": "Field 'message' is required and must be a non-empty string."}), 400
        result = predict_intent(message)
        return jsonify(result)
    except Exception as exc:  # noqa: BLE001 - surface error to client for debugging
        return jsonify({"error": f"internal error: {exc}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
