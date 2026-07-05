import os
import json
import sys
import pandas as pd
import torch

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QLabel,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QListWidget,
    QMessageBox,
    QGroupBox
)

from PyQt5.QtCore import Qt
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MAX_LEN = 64


class SentimentApp(QWidget):
    def __init__(self):
        super().__init__()

        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_map = {
            0: "Negative",
            1: "Positive"
        }

        self.tweets = []

        self.setWindowTitle("BERT Twitter Sentiment Analysis")
        self.setGeometry(200, 100, 900, 650)

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        title = QLabel("BERT Twitter Sentiment Analysis")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 22px; font-weight: bold; margin: 10px;")
        main_layout.addWidget(title)

        button_layout = QHBoxLayout()

        self.load_model_button = QPushButton("Load Model")
        self.load_model_button.clicked.connect(self.load_model)
        button_layout.addWidget(self.load_model_button)

        self.load_dataset_button = QPushButton("Load Dataset")
        self.load_dataset_button.clicked.connect(self.load_dataset)
        button_layout.addWidget(self.load_dataset_button)

        main_layout.addLayout(button_layout)

        self.status_label = QLabel("Status: Please load model and dataset.")
        self.status_label.setStyleSheet("font-size: 14px; color: blue;")
        main_layout.addWidget(self.status_label)

        dataset_group = QGroupBox("Loaded Tweets / Sentences")
        dataset_layout = QVBoxLayout()

        self.tweet_list = QListWidget()
        self.tweet_list.itemClicked.connect(self.predict_selected_tweet)
        dataset_layout.addWidget(self.tweet_list)

        dataset_group.setLayout(dataset_layout)
        main_layout.addWidget(dataset_group)

        manual_group = QGroupBox("Manual Sentiment Prediction")
        manual_layout = QVBoxLayout()

        self.manual_input = QTextEdit()
        self.manual_input.setPlaceholderText("Type a sentence here...")
        self.manual_input.setFixedHeight(90)
        manual_layout.addWidget(self.manual_input)

        self.predict_button = QPushButton("Predict Manual Text")
        self.predict_button.clicked.connect(self.predict_manual_text)
        manual_layout.addWidget(self.predict_button)

        manual_group.setLayout(manual_layout)
        main_layout.addWidget(manual_group)

        self.result_label = QLabel("Prediction Result: Not predicted yet.")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; padding: 12px; color: green;"
        )
        main_layout.addWidget(self.result_label)

        self.setLayout(main_layout)

    def load_label_map(self, model_folder):
        label_map_path = os.path.join(model_folder, "label_map.json")

        if os.path.exists(label_map_path):
            with open(label_map_path, "r") as f:
                label_map = json.load(f)

            self.label_map = {int(k): v for k, v in label_map.items()}

        elif self.model is not None and hasattr(self.model.config, "id2label"):
            self.label_map = {
                int(k): v for k, v in self.model.config.id2label.items()
            }

        else:
            self.label_map = {
                0: "Negative",
                1: "Positive"
            }

    def load_model(self):
        model_folder = QFileDialog.getExistingDirectory(
            self,
            "Select Saved BERT Model Folder"
        )

        if not model_folder:
            return

        try:
            self.status_label.setText("Status: Loading model, please wait...")
            QApplication.processEvents()

            self.tokenizer = AutoTokenizer.from_pretrained(model_folder)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_folder)
            self.model = self.model.to(self.device)
            self.model.eval()

            self.load_label_map(model_folder)

            self.status_label.setText(f"Status: Model loaded successfully from {model_folder}")

        except Exception as e:
            QMessageBox.critical(self, "Model Loading Error", str(e))
            self.status_label.setText("Status: Model loading failed.")

    def load_dataset(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Twitter Sentiment CSV File",
            "",
            "CSV Files (*.csv)"
        )

        if not file_path:
            return

        try:
            df = pd.read_csv(file_path)

            if "text" not in df.columns:
                QMessageBox.warning(
                    self,
                    "Dataset Error",
                    "Dataset must contain a column named 'text'."
                )
                return

            df = df.dropna(subset=["text"])
            df["text"] = df["text"].astype(str)

            self.tweets = df["text"].tolist()

            self.tweet_list.clear()

            for tweet in self.tweets:
                self.tweet_list.addItem(tweet)

            self.status_label.setText(
                f"Status: Dataset loaded successfully. Total tweets: {len(self.tweets)}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Dataset Loading Error", str(e))
            self.status_label.setText("Status: Dataset loading failed.")

    def predict_sentiment(self, text):
        if self.model is None or self.tokenizer is None:
            QMessageBox.warning(
                self,
                "Model Not Loaded",
                "Please load the saved BERT model first."
            )
            return None, None

        if not text.strip():
            QMessageBox.warning(
                self,
                "Empty Text",
                "Please enter or select a sentence."
            )
            return None, None

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            probabilities = torch.softmax(outputs.logits, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)

        predicted_label = prediction.item()
        predicted_sentiment = self.label_map.get(predicted_label, str(predicted_label))
        confidence_score = confidence.item() * 100

        return predicted_sentiment, confidence_score

    def predict_selected_tweet(self, item):
        selected_text = item.text()

        sentiment, confidence = self.predict_sentiment(selected_text)

        if sentiment is not None:
            self.result_label.setText(
                f"Prediction Result: {sentiment} | Confidence: {confidence:.2f}%"
            )

    def predict_manual_text(self):
        text = self.manual_input.toPlainText()

        sentiment, confidence = self.predict_sentiment(text)

        if sentiment is not None:
            self.result_label.setText(
                f"Prediction Result: {sentiment} | Confidence: {confidence:.2f}%"
            )


def main():
    app = QApplication(sys.argv)
    window = SentimentApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()