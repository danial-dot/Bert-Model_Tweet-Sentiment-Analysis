# BERT Twitter Sentiment Analysis with PyQt GUI

## Project Overview
This project is a Twitter sentiment analysis system using a trained BERT model. The model is trained on a Twitter sentiment dataset and connected with a PyQt desktop GUI. The application allows the user to load the trained model, load a dataset, select a tweet, and predict whether the sentiment is Positive or Negative. It also allows manual text input for sentiment prediction.

## Assignment Topic
BERT Training on Twitter Sentiment Dataset with PyQt GUI

## Dataset
The dataset used in this project is a Twitter sentiment dataset in CSV format.

The original dataset contains the following useful columns:

- `text`: tweet text
- `polarity`: original sentiment label

The original polarity labels were converted as follows:

| Original Polarity | Converted Label | Sentiment |
|---|---|---|
| 0 | 0 | Negative |
| 4 | 1 | Positive |

A balanced dataset was prepared by selecting an equal number of positive and negative tweets.

## Model Details
The model used in this project is:

```text
bert-base-uncased

Classes:

Negative
Positive
```
#Project Structure
Assignment3_Bert_Sentiment_Analysis/
|-- Assignment.py
|-- train_bert.py
|-- test_model.py
|-- app.py
|-- requirements.txt
|-- README.md
|-- report.pdf
|-- dataset/
|   |-- twitter_sentiment_clean.csv
|-- results/
|   |-- metrics.json
|   |-- confusion_matrix.png
|   |-- training_validation_loss.png
|   |-- training_validation_accuracy.png
|-- screenshots/
|   |-- gui_home.png
|   |-- model_loaded.png
|   |-- dataset_loaded.png
|   |-- prediction_result.png

#GUI Features
The PyQt GUI includes:

Load Model button
Load Dataset button
Tweet list display
Clickable tweet prediction
Manual text input
Predict button
Prediction result display
Status message

#Results
The trained model was evaluated using accuracy, precision, recall, F1-score, and confusion matrix.

Metric	Value
Accuracy	[0.7775]
Precision	[0.7952]
Recall	[0.7775]
F1-score	[0.7741]

#Saved Model
The trained BERT model files are large, so the saved model folder is uploaded here with the name "Train_bert.py.

#Conclusion
This project successfully trains a BERT model for Twitter sentiment analysis and connects it with a PyQt GUI. The final application can load the saved model, load a dataset, predict sentiment from selected tweets, and predict sentiment from manually entered text.


