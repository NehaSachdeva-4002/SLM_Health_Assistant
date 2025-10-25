from flask import Flask, render_template, request, jsonify
from model.slm_model import SLMModel
from utils.preprocess import preprocess_text

app = Flask(__name__)
app.config.from_object('config.Config')

# Create a single model instance (stub) — replace with actual load in production
model = SLMModel()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/generate', methods=['POST'])
def generate():
    data = request.json or {}
    prompt = data.get('prompt', '')
    prompt = preprocess_text(prompt)
    # model.generate is a stub; replace with your model invocation
    output = model.generate(prompt)
    return jsonify({'prompt': prompt, 'output': output})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import torch
import os

app = Flask(__name__)

# Load fine-tuned/pretrained model from fine_tuning.py output
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "final_model")

# Ensure model directory exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}"
    )

# Setup device (GPU if available)
device = 0 if torch.cuda.is_available() else -1
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# Initialize emotion classifier pipeline
emotion_classifier = pipeline(
    "text-classification",
    model=MODEL_PATH,
    device=device,
    top_k=None
)

def predict_emotion(text: str):
    """Get emotion predictions for given text."""
    results = emotion_classifier(text)
    top_result = max(results[0], key=lambda x: x["score"])
    return {
        "emotion": top_result["label"],
        "confidence": round(top_result["score"] * 100, 2),
        "all_predictions": sorted(results[0], key=lambda x: x["score"], reverse=True)
    }

# API endpoint for predictions
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400
    text = data["text"]
    result = predict_emotion(text)
    return jsonify(result)

# Optional web interface
@app.route("/", methods=["GET", "POST"])
def home():
    emotion_result = None
    if request.method == "POST":
        text = request.form["text"]
        emotion_result = predict_emotion(text)
    return render_template("index.html", result=emotion_result)

if __name__ == "__main__":
    print("\n Flask Emotion Classifier API starting...")
    print("Model loaded from:", MODEL_PATH)
    app.run(host="0.0.0.0", port=5000, debug=True)
