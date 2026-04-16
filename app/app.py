"""
Hybrid Medical Chatbot - Streamlit Application
LSTM Intent Detection + Groq LLM + OpenFDA Drug Facts
"""

import streamlit as st
import sys
from pathlib import Path

# ---------------- PATH SETUP ----------------
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from predict import IntentPredictor
from llm_fallback import LLMFallback


# ---------------- CONFIG ----------------
CONFIDENCE_THRESHOLD = 0.85


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Hybrid AI Medical Assistant",
    page_icon="🏥",
    layout="centered",
)


# ---------------- LOAD MODELS (CACHED) ----------------
@st.cache_resource
def load_models():
    predictor = IntentPredictor()
    llm = LLMFallback()
    return predictor, llm


# ---------------- SESSION INIT ----------------
def init_session():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "predictor" not in st.session_state:
        st.session_state.predictor, st.session_state.llm = load_models()


# ---------------- HYBRID RESPONSE ENGINE ----------------
def get_bot_response(user_input: str):
    predictor = st.session_state.predictor
    llm = st.session_state.llm

    # Predict intent using LSTM
    tag, confidence = predictor.predict_intent_with_confidence(user_input)

    # Route decision
    if confidence < CONFIDENCE_THRESHOLD or tag == "unknown":
        response = llm.medical_answer(user_input)   # 🔥 AI + OpenFDA
        source = "llm"
    else:
        response = predictor.get_predefined_response(tag)  # 🧠 LSTM
        source = "lstm"

    return response, tag, confidence, source


# ---------------- CHAT DISPLAY ----------------
def display_chat():
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])

        else:
            badge = "🧠 LSTM" if msg["source"] == "lstm" else "🤖 AI + OpenFDA"

            st.chat_message("assistant").write(
                f"{msg['content']}\n\n"
                f"*{badge} | Confidence: {msg['confidence']:.2f} | Intent: {msg['intent']}*"
            )


# ---------------- MAIN APP ----------------
def main():
    init_session()

    # Header
    st.title("🏥 Hybrid AI Medical Assistant")
    st.caption("Deep Learning Intent Detection + AI Medical Knowledge")

    # Medical disclaimer
    st.warning(
        "⚠️ This is **NOT** a medical diagnosis tool. "
        "Always consult a qualified healthcare professional."
    )

    # Show previous chat
    display_chat()

    # Chat input
    user_input = st.chat_input("Describe symptoms or ask a medical question...")

    if user_input:
        # Save user message
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input}
        )

        # Get hybrid response
        with st.spinner("🧠 Analyzing with Hybrid AI..."):
            response, intent, confidence, source = get_bot_response(user_input)

        # Save bot message
        st.session_state.chat_history.append(
            {
                "role": "bot",
                "content": response,
                "intent": intent,
                "confidence": confidence,
                "source": source,
            }
        )

        # Refresh UI
        st.rerun()


# ---------------- RUN ----------------
if __name__ == "__main__":
    main()
