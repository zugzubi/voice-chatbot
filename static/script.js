/* Voice-Enabled Chatbot frontend
 * - Web Speech API for speech recognition
 * - SpeechSynthesis for text-to-speech
 * - POST /predict to the Flask backend
 */

const chatWindow = document.getElementById("chatWindow");
const micBtn = document.getElementById("micBtn");
const micLabel = document.getElementById("micLabel");
const textInput = document.getElementById("textInput");
const sendBtn = document.getElementById("sendBtn");
const ttsBtn = document.getElementById("ttsBtn");
const statusPill = document.getElementById("statusPill");
const statusText = document.getElementById("statusText");
const techPanel = document.getElementById("techPanel");
const techIntent = document.getElementById("techIntent");
const techConfidence = document.getElementById("techConfidence");
const techSpeech = document.getElementById("techSpeech");

let lastBotResponse = "";

function appendMessage(text, sender) {
    const wrap = document.createElement("div");
    wrap.className = `msg ${sender}`;
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;
    wrap.appendChild(bubble);
    chatWindow.appendChild(wrap);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

function setStatus(state, text) {
    statusPill.classList.remove("listening", "thinking");
    if (state) statusPill.classList.add(state);
    statusText.textContent = text;
}

function speak(text) {
    if (!("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    const utter = new SpeechSynthesisUtterance(text);
    utter.rate = 1.0;
    utter.pitch = 1.0;
    window.speechSynthesis.speak(utter);
}

async function sendMessage(message, opts = {}) {
    const msg = (message || "").trim();
    if (!msg) return;
    appendMessage(msg, "user");
    textInput.value = "";
    setStatus("thinking", "Thinking...");

    try {
        const res = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: msg }),
        });
        const data = await res.json();
        if (!res.ok) {
            appendMessage(`Error: ${data.error || "request failed"}`, "bot");
            setStatus(null, "Ready");
            return;
        }

        appendMessage(data.response, "bot");
        lastBotResponse = data.response;

        techPanel.hidden = false;
        techIntent.textContent = data.intent;
        techConfidence.textContent = `${(data.confidence * 100).toFixed(1)}%`;
        if (opts.spokenText) techSpeech.textContent = opts.spokenText;

        speak(data.response);
        setStatus(null, "Ready");
    } catch (err) {
        appendMessage(`Network error: ${err.message}`, "bot");
        setStatus(null, "Ready");
    }
}

// ---- text input ----
sendBtn.addEventListener("click", () => sendMessage(textInput.value));
textInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendMessage(textInput.value);
});
ttsBtn.addEventListener("click", () => { if (lastBotResponse) speak(lastBotResponse); });

// ---- speech recognition ----
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;
let listening = false;

if (SpeechRecognition) {
    recognition = new SpeechRecognition();
    recognition.lang = "en-US";
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
        listening = true;
        micBtn.classList.add("listening");
        micLabel.textContent = "Listening...";
        setStatus("listening", "Listening...");
    };
    recognition.onerror = (e) => {
        listening = false;
        micBtn.classList.remove("listening");
        micLabel.textContent = "Start Speaking";
        setStatus(null, `Mic error: ${e.error}`);
    };
    recognition.onend = () => {
        listening = false;
        micBtn.classList.remove("listening");
        micLabel.textContent = "Start Speaking";
    };
    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        techSpeech.textContent = transcript;
        sendMessage(transcript, { spokenText: transcript });
    };
} else {
    micBtn.disabled = true;
    micLabel.textContent = "Mic not supported";
    micBtn.title = "Your browser does not support the Web Speech API. Try Chrome or Edge.";
}

micBtn.addEventListener("click", () => {
    if (!recognition) return;
    if (listening) {
        recognition.stop();
    } else {
        try { recognition.start(); } catch (_) { /* already started */ }
    }
});
