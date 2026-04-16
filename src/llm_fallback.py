import os
import requests
from groq import Groq
from dotenv import load_dotenv
from pathlib import Path

# ---------- LOAD ENV FROM PROJECT ROOT ----------
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class LLMFallback:
    """
    Hybrid LLM fallback using:
    - Groq LLM for explanation
    - OpenFDA API for factual drug info
    """

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            print("⚠️ GROQ API key NOT found in .env")
            self.client = None
        else:
            try:
                self.client = Groq(api_key=api_key)
                print("✅ Groq LLM initialized successfully")
            except Exception as e:
                print("🚨 Groq initialization error:", e)
                self.client = None

    # ---------- CHECK AVAILABILITY ----------
    def is_available(self):
        return self.client is not None

    # ---------- GROQ LLM RESPONSE ----------
    def ask_llm(self, question: str) -> str:
        """
        Calls Groq LLM for general medical explanation.
        """
        if not self.client:
            return "AI service unavailable. Please consult a healthcare professional."

        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a safe medical assistant. "
                            "Provide general medical information only. "
                            "Do NOT prescribe medicines or dosage. "
                            "Always recommend consulting a doctor."
                        ),
                    },
                    {"role": "user", "content": question},
                ],
                temperature=0.3,
                max_tokens=300,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print("🚨 GROQ CALL ERROR:", e)
            return "I couldn't process that. Please consult a healthcare professional."

    # ---------- OPENFDA DRUG LOOKUP ----------
    def get_drug_info(self, drug: str) -> str:
        """
        Fetches official drug purpose & warnings from OpenFDA.
        """
        try:
            url = (
                "https://api.fda.gov/drug/label.json"
                f"?search=openfda.generic_name:{drug}&limit=1"
            )

            r = requests.get(url, timeout=5)

            if r.status_code != 200:
                return ""

            data = r.json()["results"][0]

            purpose = data.get("purpose", ["No purpose info available."])[0]
            warnings = data.get("warnings", ["No warnings info available."])[0]

            return (
                "\n\n📄 **OpenFDA Drug Information**\n"
                f"**Purpose:** {purpose}\n"
                f"**Warnings:** {warnings}"
            )

        except Exception:
            return ""

    # ---------- COMBINED MEDICAL RESPONSE ----------
    def medical_answer(self, question: str) -> str:
        """
        Returns:
        - AI explanation from Groq
        - + OpenFDA drug facts if detected
        """
        ai_text = self.ask_llm(question)

        # Detect possible drug names in question
        words = question.lower().replace("?", "").split()

        for word in words:
            drug_info = self.get_drug_info(word)
            if drug_info:
                return ai_text + drug_info

        return ai_text


# ---------- QUICK TERMINAL TEST ----------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Testing Groq + OpenFDA Hybrid LLM")
    print("=" * 60)

    llm = LLMFallback()

    if not llm.is_available():
        print("❌ Groq not available. Check GROQ_API_KEY in .env")
    else:
        test_question = "What is paracetamol used for?"
        print(f"\nQuestion: {test_question}\n")
        print("Response:\n")
        print(llm.medical_answer(test_question))
