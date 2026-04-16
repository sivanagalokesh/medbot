# 🏥 MedBot – AI-Powered Medical Chatbot

## 🚀 Overview

MedBot is an AI-powered medical chatbot designed to provide preliminary health guidance based on user symptoms. It leverages deep learning and natural language processing to understand user queries and return relevant medical responses.

⚠️ **Disclaimer:** This system is for educational purposes only and does not replace professional medical advice.

---

## 🎯 Problem Statement

Access to quick and reliable preliminary health information is limited. Many users rely on unreliable sources or delay seeking medical help.

👉 MedBot aims to:

* Provide instant symptom-based responses
* Assist users with basic health guidance
* Demonstrate AI applications in healthcare

---

## 🧠 Model & Approach

### 🔹 Model Type

* Deep Learning-based Text Classification
* Trained using TensorFlow / Keras

### 🔹 Workflow

1. Data preprocessing (tokenization, cleaning)
2. Label encoding
3. Model training
4. Prediction using trained model
5. Fallback using LLM (if confidence is low)

### 🔹 Components

* `tokenizer.pkl` → Text vectorization
* `label_encoder.pkl` → Output decoding
* `model.h5` → Trained model

---

## 🏗️ Project Structure

```
medbot/
│
├── app/                # Flask application
│   ├── app.py
│   └── api.py
│
├── src/                # Core ML logic
│   ├── train.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── evaluate.py
│   └── utils.py
│
├── models/             # Saved ML models
│
├── data/               # Dataset (intents)
│
├── requirements.txt    # Dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ Tech Stack

* Python 🐍
* TensorFlow / Keras 🤖
* Flask 🌐
* NumPy & Pandas
* NLP techniques

---

## 💡 Features

✅ Symptom-based query handling
✅ Deep learning model for classification
✅ Modular ML pipeline
✅ Flask API integration
✅ LLM fallback support (advanced feature)

---

## 📸 Demo

### Example Interaction:

```
User: I have headache and fever
Bot: You may be experiencing a viral infection. Please consult a doctor if symptoms persist.
```

---

## 🏃‍♂️ How to Run Locally

### 1. Clone the repository

```
git clone https://github.com/sivanagalokesh/medbot.git
cd medbot
```

### 2. Create virtual environment

```
python -m venv myenv
myenv\Scripts\activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Run the application

```
python app/app.py
```

### 5. Open in browser

```
http://127.0.0.1:5000/
```

---

## 📊 Future Improvements

* 🔹 Improve model accuracy with larger dataset
* 🔹 Add voice-based interaction
* 🔹 Integrate real-time medical APIs
* 🔹 Deploy on cloud (Render / AWS)
* 🔹 Add user authentication & history

---

## ⚠️ Disclaimer

This chatbot is not a substitute for professional medical advice. Always consult a qualified healthcare provider for diagnosis and treatment.

---

## 👨‍💻 Author

**Siva Naga Lokesh B**
🔗 GitHub: https://github.com/sivanagalokesh

---

## ⭐ If you like this project

Give it a star ⭐ and share it!

---
