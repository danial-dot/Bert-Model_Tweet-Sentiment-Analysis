import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_PATH = "saved_bert_model"
MAX_LEN = 64


def load_label_map(model_path, model):
    label_map_path = os.path.join(model_path, "label_map.json")

    if os.path.exists(label_map_path):
        with open(label_map_path, "r") as f:
            label_map = json.load(f)

        return {int(k): v for k, v in label_map.items()}

    if hasattr(model.config, "id2label"):
        return {int(k): v for k, v in model.config.id2label.items()}

    return {
        0: "Negative",
        1: "Positive"
    }


def predict_sentiment(text, tokenizer, model, label_map, device):
    model.eval()

    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        probabilities = torch.softmax(outputs.logits, dim=1)
        confidence, prediction = torch.max(probabilities, dim=1)

    predicted_label = prediction.item()
    predicted_sentiment = label_map[predicted_label]
    confidence_score = confidence.item() * 100

    return predicted_sentiment, confidence_score


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model folder not found: {MODEL_PATH}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading saved model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model = model.to(device)

    label_map = load_label_map(MODEL_PATH, model)

    print("Model loaded successfully.")
    print("Label map:", label_map)

    sample_sentences = [
        "I am very happy today!",
        "This is the worst day of my life.",
        "I really enjoyed this product.",
        "I hate this experience."
    ]

    print("\nSample Predictions:")
    for sentence in sample_sentences:
        sentiment, confidence = predict_sentiment(
            sentence,
            tokenizer,
            model,
            label_map,
            device
        )
        print(f"Text: {sentence}")
        print(f"Prediction: {sentiment} ({confidence:.2f}%)")
        print("-" * 50)

    while True:
        user_text = input("\nEnter a sentence for prediction or type exit: ")

        if user_text.lower() == "exit":
            break

        sentiment, confidence = predict_sentiment(
            user_text,
            tokenizer,
            model,
            label_map,
            device
        )

        print(f"Prediction: {sentiment}")
        print(f"Confidence: {confidence:.2f}%")


if __name__ == "__main__":
    main()