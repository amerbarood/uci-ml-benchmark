"""SMS spam detection with TF-IDF features and Multinomial Naive Bayes.

Usage:
    python src/spam_detection.py --data data/text/SMSSpamCollection
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/text/SMSSpamCollection")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    df = pd.read_csv(args.data, sep="\t", names=["label", "message"])
    y = df["label"].map({"ham": 0, "spam": 1})
    X_tr, X_te, y_tr, y_te = train_test_split(df["message"], y, test_size=0.2,
                                              random_state=42, stratify=y)

    vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
    model = MultinomialNB().fit(vectorizer.fit_transform(X_tr), y_tr)
    pred = model.predict(vectorizer.transform(X_te))

    print(f"Accuracy: {accuracy_score(y_te, pred):.2%}")
    print(classification_report(y_te, pred, target_names=["Ham", "Spam"]))

    os.makedirs(os.path.join(args.out, "figures"), exist_ok=True)
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_te, pred), annot=True, fmt="d", cmap="Blues",
                xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("SMS spam detection")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "figures", "sms_spam_cm.png"), dpi=120)


if __name__ == "__main__":
    main()
