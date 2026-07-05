# Bert-Model_Tweet-Sentiment-Analysis
This project performs Twitter sentiment analysis using a trained BERT model. The model is trained on a Twitter sentiment dataset and connected with a PyQt desktop GUI. The user can load the saved model, load a dataset, click any tweet, or type a sentence manually to predict sentiment.
# Dataset Dataset used
Sentiment140 / Twitter Sentiment Dataset The dataset contains Twitter text and polarity labels. Labels used: - 0 = Negative - 1 = Positive Original polarity mapping: - 0 = Negative - 4 = Positive Only the useful columns `text` and `label` were used for model training.
# Model Model used 
bert-base-uncased Task: Binary sentiment classification Classes: - Negative - Positive
