# SLM Flask App (starter)

Minimal Flask app scaffold for a Small Language Model (SLM) web frontend.
<img width="873" height="868" alt="Screenshot 2025-10-25 232954" src="https://github.com/user-attachments/assets/87c0f960-a391-4573-97a7-465e7935aab0" />
<img width="866" height="868" alt="Screenshot 2025-10-25 233031" src="https://github.com/user-attachments/assets/907aa19b-724a-4f88-a7af-11ae029cb931" />
<img width="677" height="206" alt="Screenshot 2025-10-25 233101" src="https://github.com/user-attachments/assets/25cd75ae-19a5-4876-b394-503e2e03f9a5" />




Quick start

1. Create and activate a virtualenv:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the app:

```powershell
python run.py
```

Files created:
- `app.py` - Flask application
- `run.py` - development runner
- `model/slm_model.py` - model loading stub
- `utils/preprocess.py` - preprocessing helpers
- `templates/index.html` - basic UI

Next steps: integrate your model code in `model/slm_model.py`, add tests, and (optionally) add Docker/GitHub Actions.
🧠 SLM Flask App
A lightweight, efficient Small Language Model (SLM) Flask application for emotion detection — designed for mental health and emotional understanding use cases.

🚀 Overview
Large Language Models (LLMs) like GPT‑4 have revolutionized AI, but in specialized domains, bigger isn’t always better.

Small Language Models (SLMs) offer the perfect balance of precision, speed, and privacy.
This project demonstrates how an SLM‑powered assistant, fine-tuned from DistilBERT, can accurately detect emotions with minimal resources — making it ideal for real‑world, privacy-aware, or on-device applications.

💡 Why Use SLMs
🔍 Task‑Specific Precision
Fine‑tuning on emotion datasets like EmpatheticDialogues allows SLMs to capture emotional nuance that general-purpose LLMs often miss.

⚡ Efficiency & Speed
Smaller models require less compute — enabling real‑time inference even on modest hardware or local machines.

🔐 Privacy by Design
Sensitive mental health data never leaves your system, making this architecture ideal for local or enterprise use.

💰 Cost‑Effective
Lower compute and memory needs make SLMs more accessible to startups, researchers, and nonprofits.

🧩 About the Project
This AI‑powered Emotion Detection Assistant classifies text into key emotional states such as:

Joy

Sadness

Anger

Fear

Neutrality

Core Features
Flask‑based backend for fast REST APIs

Supports GPU acceleration via PyTorch

Clean, responsive web interface for real‑time emotional analysis

Modular directory structure for extendability

⚙️ Setup and Installation
1. Clone the repository
bash
git clone https://github.com/your-username/slm-flask-app.git
cd slm-flask-app
2. Create a virtual environment
bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
3. Install dependencies
bash
pip install -r requirements.txt
4. Run the Flask app
bash
python run.py
The app will run locally at:

text
http://127.0.0.1:5000/
🧠 Model Details
The model used is a fine‑tuned DistilBERT trained for emotion classification.

Saved under model/final_model/

Tokenizer included in the same directory

Easily replaceable with your custom fine‑tuned model

For customization:

python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("model/final_model")
model = AutoModelForSequenceClassification.from_pretrained("model/final_model")
📁 Project Structure
text
slm_flask_app/
│
├── app.py                # Main Flask application
├── fine_tuning.py        # Model preparation and fine‑tuning logic
├── model/
│   └── final_model/      # Saved model + tokenizer
├── templates/            # HTML templates for UI
├── static/               # CSS / JS / assets
└── requirements.txt      # Project dependencies
🧩 Example Use Case
Enter a user statement like:

“Lately, I’ve been feeling really overwhelmed and anxious about everything happening around me.”

The model returns:

json
{
  "emotion": "Fear",
  "confidence": 0.91,
  "all_predictions": [
    {"label": "Fear", "score": 0.91},
    {"label": "Sadness", "score": 0.07},
    {"label": "Anger", "score": 0.02}
  ]
}
🌱 Future Enhancements
Add support for multilingual emotion detection

Integrate speech-to-text emotion analysis

Extend model to empathy and tone classification

🧾 License
This project is released under the MIT License.
Feel free to use, modify, and build upon it for educational or research purposes.
