import os
import json
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report




DATASET_PATH = "E:/Assignment3_Bert_Sentiment_Analysis/dataset/twitter_sentiment_clean.csv"
MODEL_NAME = "bert-base-uncased"
SAVE_MODEL_PATH = "saved_bert_model"
RESULTS_PATH = "results"

MAX_LEN = 64
BATCH_SIZE = 8
EPOCHS = 1
LEARNING_RATE = 2e-5

LABEL_NAMES = {
    0: "Negative",
    1: "Positive"
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TwitterSentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        label = int(self.labels[index])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long)
        }


def load_data():
    df = pd.read_csv(DATASET_PATH)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns.")

    df = df[["text", "label"]].dropna()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    print("Dataset loaded successfully.")
    print(df.head())
    print("\nClass count:")
    print(df["label"].value_counts())

    return df


def train_one_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_examples = 0

    for batch in data_loader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        _, preds = torch.max(logits, dim=1)

        correct_predictions += torch.sum(preds == labels).item()
        total_examples += labels.size(0)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_examples

    return avg_loss, accuracy


def evaluate_model(model, data_loader, device):
    model.eval()

    total_loss = 0
    predictions = []
    actual_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            _, preds = torch.max(logits, dim=1)

            total_loss += loss.item()
            predictions.extend(preds.cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(actual_labels, predictions)

    return avg_loss, accuracy, actual_labels, predictions


def save_training_graphs(train_losses, val_losses, train_accuracies, val_accuracies):
    os.makedirs(RESULTS_PATH, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, "training_validation_loss.png"))
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(train_accuracies, label="Training Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, "training_validation_accuracy.png"))
    plt.close()


def save_confusion_matrix(y_true, y_pred):
    os.makedirs(RESULTS_PATH, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"]
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, "confusion_matrix.png"))
    plt.close()


def main():
    set_seed(42)

    os.makedirs(SAVE_MODEL_PATH, exist_ok=True)
    os.makedirs(RESULTS_PATH, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = load_data()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    print("\nLoading tokenizer and BERT model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "Negative", 1: "Positive"},
        label2id={"Negative": 0, "Positive": 1}
    )

    model = model.to(device)

    train_dataset = TwitterSentimentDataset(
        train_texts,
        train_labels,
        tokenizer,
        MAX_LEN
    )

    val_dataset = TwitterSentimentDataset(
        val_texts,
        val_labels,
        tokenizer,
        MAX_LEN
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    print("\nTraining started...")

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device
        )

        val_loss, val_acc, y_true, y_pred = evaluate_model(
            model,
            val_loader,
            device
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Training Loss: {train_loss:.4f}")
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted"
    )

    final_accuracy = accuracy_score(y_true, y_pred)

    print("\nFinal Evaluation Results")
    print(f"Accuracy:  {final_accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["Negative", "Positive"]
        )
    )

    save_training_graphs(
        train_losses,
        val_losses,
        train_accuracies,
        val_accuracies
    )

    save_confusion_matrix(y_true, y_pred)

    print("\nSaving trained model and tokenizer...")

    model.save_pretrained(SAVE_MODEL_PATH)
    tokenizer.save_pretrained(SAVE_MODEL_PATH)

    with open(os.path.join(SAVE_MODEL_PATH, "label_map.json"), "w") as f:
        json.dump(LABEL_NAMES, f, indent=4)

    metrics = {
        "accuracy": final_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "model_name": MODEL_NAME
    }

    with open(os.path.join(RESULTS_PATH, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nTraining completed successfully.")
    print(f"Model saved in: {SAVE_MODEL_PATH}")
    print(f"Graphs and metrics saved in: {RESULTS_PATH}")


if __name__ == "__main__":
    main()