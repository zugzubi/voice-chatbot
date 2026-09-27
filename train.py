"""
Training script for the Voice-Enabled Chatbot intent classifier.

Loads data/intents.json, trains a BiLSTM neural network, and saves:
    models/chatbot_model.keras
    models/tokenizer.pkl
    models/label_encoder.pkl
    models/config.pkl        (max_len, threshold, etc.)
    models/metrics.json      (final training/validation/test metrics)
"""

from __future__ import annotations

import json
import os
import pickle
import random

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Bidirectional,
    Dense,
    Dropout,
    Embedding,
    GlobalMaxPooling1D,
    LSTM,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_PATH = os.path.join("data", "intents.json")
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

VOCAB_SIZE = 2000
EMBEDDING_DIM = 64
MAX_LEN = 20
EPOCHS = 200
BATCH_SIZE = 16
CONFIDENCE_THRESHOLD = 0.50


def load_data(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts, labels = [], []
    for intent in data["intents"]:
        tag = intent["tag"]
        for pattern in intent["patterns"]:
            texts.append(pattern.lower().strip())
            labels.append(tag)
    return texts, labels


def build_model(vocab_size: int, num_classes: int) -> tf.keras.Model:
    model = Sequential(
        [
            Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=MAX_LEN, mask_zero=True),
            Bidirectional(LSTM(64, return_sequences=True)),
            GlobalMaxPooling1D(),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    print(f"[train] TensorFlow version: {tf.__version__}")
    print(f"[train] Loading dataset from {DATA_PATH}")
    texts, labels = load_data(DATA_PATH)
    print(f"[train] Total training sentences: {len(texts)}")
    print(f"[train] Unique intents: {len(set(labels))}")

    # Tokenizer
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")

    # Label encoding
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    print(f"[train] Classes: {list(label_encoder.classes_)}")

    # Train/val/test split (stratified where possible)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )
    print(f"[train] Train / Val / Test = {len(X_train)} / {len(X_val)} / {len(X_test)}")

    # Model
    actual_vocab = min(VOCAB_SIZE, len(tokenizer.word_index) + 1)
    model = build_model(actual_vocab, num_classes)
    model.summary()

    early_stop = EarlyStopping(
        monitor="val_accuracy", patience=25, restore_best_weights=True, verbose=1
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=2,
    )

    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    print("\n[train] Final metrics")
    print(f"  Train      -> loss={train_loss:.4f}  acc={train_acc:.4f}")
    print(f"  Validation -> loss={val_loss:.4f}  acc={val_acc:.4f}")
    print(f"  Test       -> loss={test_loss:.4f}  acc={test_acc:.4f}")

    # Save artifacts
    model_path = os.path.join(MODELS_DIR, "chatbot_model.keras")
    model.save(model_path)
    with open(os.path.join(MODELS_DIR, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)
    with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)
    with open(os.path.join(MODELS_DIR, "config.pkl"), "wb") as f:
        pickle.dump(
            {
                "max_len": MAX_LEN,
                "vocab_size": actual_vocab,
                "confidence_threshold": CONFIDENCE_THRESHOLD,
            },
            f,
        )

    metrics = {
        "train_loss": float(train_loss),
        "train_accuracy": float(train_acc),
        "val_loss": float(val_loss),
        "val_accuracy": float(val_acc),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "epochs_run": len(history.history["loss"]),
        "num_classes": int(num_classes),
        "num_samples": int(len(texts)),
        "classes": list(label_encoder.classes_),
    }
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[train] Saved model and artifacts to '{MODELS_DIR}/'")
    print("[train] Done.")


if __name__ == "__main__":
    main()
