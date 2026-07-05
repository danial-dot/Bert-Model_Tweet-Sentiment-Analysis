import os
import pandas as pd
import matplotlib.pyplot as plt



INPUT_FILE = "dataset/twitter_sentiment.csv"
OUTPUT_FILE = "dataset/twitter_sentiment_clean.csv"
GRAPH_FILE = "screenshots/class_distribution.png"

SAMPLES_PER_CLASS = 1000 


def load_dataset(file_path):
    """
    E:/Assignment3_Bert_Sentiment_Analysis/dataset/Twitter_Sentiment.csv
    """

    try:
        df = pd.read_csv(file_path, encoding="latin-1")
    except Exception:
        df = pd.read_csv(file_path, encoding="ISO-8859-1")

    # If dataset has no proper column names, assign Sentiment140 columns
    if "polarity" not in df.columns or "text" not in df.columns:
        if df.shape[1] >= 6:
            df = pd.read_csv(
                file_path,
                encoding="latin-1",
                header=None,
                names=["polarity", "id", "created", "query", "user", "text"]
            )

    return df


def clean_dataset(df):
    """
    Keeps only text and polarity columns.
    Converts polarity labels into model labels.
    """

    df = df[["text", "polarity"]]

    df = df.dropna()

    df["text"] = df["text"].astype(str)
    df["text"] = df["text"].str.strip()

    df = df[df["text"] != ""]

   
    df = df[df["polarity"].isin([0, 4])]

    df["label"] = df["polarity"].map({
        0: 0,
        4: 1
    })

    df["sentiment"] = df["label"].map({
        0: "Negative",
        1: "Positive"
    })

    return df


def balance_dataset(df):
    """
    Takes equal number of positive and negative tweets.
    This avoids bias in training.
    """

    negative_df = df[df["label"] == 0]
    positive_df = df[df["label"] == 1]

    sample_size = min(SAMPLES_PER_CLASS, len(negative_df), len(positive_df))

    negative_sample = negative_df.sample(sample_size, random_state=42)
    positive_sample = positive_df.sample(sample_size, random_state=42)

    balanced_df = pd.concat([negative_sample, positive_sample])
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    return balanced_df


def save_class_distribution_graph(df):
    """
    Saves class distribution graph required for assignment report.
    """

    os.makedirs("screenshots", exist_ok=True)

    counts = df["sentiment"].value_counts()

    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar", color=["red", "green"])
    plt.title("Class Distribution of Twitter Sentiment Dataset")
    plt.xlabel("Sentiment Class")
    plt.ylabel("Number of Tweets")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(GRAPH_FILE)
    plt.close()


def main():
    os.makedirs("dataset", exist_ok=True)

    print("Loading dataset...")
    df = load_dataset(INPUT_FILE)

    print("Original columns:")
    print(df.columns)

    print("\nCleaning dataset...")
    df = clean_dataset(df)

    print("\nOriginal class count:")
    print(df["sentiment"].value_counts())

    print("\nBalancing dataset...")
    clean_df = balance_dataset(df)

    clean_df.to_csv(OUTPUT_FILE, index=False)

    save_class_distribution_graph(clean_df)

    print("\nDataset cleaned successfully.")
    print(f"Saved clean dataset at: {OUTPUT_FILE}")
    print(f"Saved class distribution graph at: {GRAPH_FILE}")

    print("\nFinal class count:")
    print(clean_df["sentiment"].value_counts())

    print("\nPreview:")
    print(clean_df.head())


if __name__ == "__main__":
    main()