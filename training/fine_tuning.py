import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import os
import time

if __name__ == "__main__":
    SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'model', 'final_model')
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("🖥️ GPU Info:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # ============================================================
    print("\n" + "=" * 70)
    print("🚀 OPTION 1: Using Pre-trained Emotion Model (RECOMMENDED)")
    print("=" * 70)

    model_name = "bhadresh-savani/distilbert-base-uncased-emotion"

    print(f"\n📥 Loading pre-trained model: {model_name}")
    print("⚡ This model is already trained and ready to use!")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Save model for Flask app
    print(f"\n💾 Saving model to {SAVE_DIR}...")
    model.save_pretrained(SAVE_DIR, safe_serialization=True)
    tokenizer.save_pretrained(SAVE_DIR)

    # Create emotion classifier pipeline
    emotion_classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        top_k=None
    )

    # Test dataset
    test_texts = [
        "I'm so happy and excited about this amazing day!",
        "This makes me really angry and frustrated, I can't believe it.",
        "I'm feeling sad and lonely, missing my friends.",
        "Oh wow, I didn't expect that at all!",
        "I'm scared about what might happen next.",
        "I love this so much, it's wonderful!"
    ]

    print("\n" + "=" * 70)
    print("🧪 Testing Pre-trained Model:")
    print("=" * 70)

    for text in test_texts:
        results = emotion_classifier(text)
        top_result = max(results[0], key=lambda x: x['score'])

        print(f"\n📝 Text: {text}")
        print(f"🎯 Emotion: {top_result['label']}")
        print(f"📊 Confidence: {top_result['score'] * 100:.1f}%")

        print("   Top predictions:")
        for res in sorted(results[0], key=lambda x: x['score'], reverse=True)[:3]:
            print(f"      - {res['label']}: {res['score'] * 100:.1f}%")
        print("-" * 70)

    # ============================================================
    print("\n" + "=" * 70)
    
    from datasets import load_dataset
    from transformers import Trainer, TrainingArguments

    print("\n📥 Loading dataset...")
    dataset = load_dataset("dair-ai/emotion")
    print(f"Labels: {dataset['train'].features['label'].names}")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    print("🔄 Tokenizing...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    small_train = tokenized_datasets["train"].shuffle(seed=42).select(range(8000))
    small_val = tokenized_datasets["validation"].shuffle(seed=42).select(range(1000))

    training_args = TrainingArguments(
        output_dir=SAVE_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=2,
        weight_decay=0.01,
        fp16=True,
        dataloader_num_workers=0,
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train,
        eval_dataset=small_val,
    )

    print("\n🚀 Fine-tuning (2 epochs, ~8000 samples)...")
    print("⏱️ Estimated time: 5-10 minutes")
    trainer.train()

    print(f"\n💾 Saving fine-tuned model...")
    model.save_pretrained(SAVE_DIR, safe_serialization=True)
    tokenizer.save_pretrained(SAVE_DIR)

    # ============================================================
    print("\n" + "=" * 70)
    print("✅ MODEL READY!")
    print(f"📁 Saved at: {SAVE_DIR}")
    print("🎯 You can now use this model in your Flask app!")
    print("=" * 70)

    # Create Flask inference helper script
    inference_script = '''
from transformers import pipeline
import torch

model_path = "model/final_model"
device = 0 if torch.cuda.is_available() else -1

emotion_classifier = pipeline(
    "text-classification",
    model=model_path,
    device=device,
    top_k=None
)

def predict_emotion(text):
    results = emotion_classifier(text)
    top_result = max(results[0], key=lambda x: x['score'])
    return {
        'emotion': top_result['label'],
        'confidence': top_result['score'],
        'all_predictions': results[0]
    }

# Example usage
text = "I'm so happy!"
result = predict_emotion(text)
print(f"Emotion: {result['emotion']}")
print(f"Confidence: {result['confidence']:.2%}")
'''

    with open(os.path.join(SAVE_DIR, 'how_to_use.py'), 'w') as f:
        f.write(inference_script)

    print("\n📝 Usage guide saved to: how_to_use.py")

    # Performance Benchmark
    print("\n⚡ Performance Benchmark:")
    test_text = "I love this!"

    for _ in range(5):  # Warmup
        _ = emotion_classifier(test_text)

    start = time.time()
    iterations = 50
    for _ in range(iterations):
        _ = emotion_classifier(test_text)
    elapsed = time.time() - start

    print(f"Average inference time: {elapsed / iterations * 1000:.2f} ms per prediction")
    print(f"Throughput: {iterations / elapsed:.1f} predictions/sec")

