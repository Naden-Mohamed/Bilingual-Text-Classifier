import os
import io
import pickle
import zipfile

import numpy as np
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

load_dotenv()
nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("omw-1.4",   quiet=True)

USER = os.getenv("KAGGLE_USERNAME")
KEY  = os.getenv("KAGGLE_API_TOKEN")

SAMPLE_SIZE  = 20_000
MAX_FEATURES = 3_000


class EnglishTextClassifier:

    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.tfidf      = None
        self.model      = None

    def get_english_data(self) -> pd.DataFrame:
        url = "https://www.kaggle.com/api/v1/datasets/download/abdelmalekeladjelet/sentiment-analysis-dataset"
        response = requests.get(url, auth=HTTPBasicAuth(USER, KEY))
        if response.status_code != 200:
            raise Exception(f"Download failed – HTTP {response.status_code}")
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            file_name = z.namelist()[0]
            with z.open(file_name) as f:
                df = pd.read_csv(f)
        df.columns = [c.strip().lower() for c in df.columns]
        df.rename(columns={"sentiment": "label", "comment": "text"}, inplace=True)
        df.dropna(subset=["text"], inplace=True)
        df.drop_duplicates(subset=["text"], inplace=True)
        df["text"] = df["text"].astype(str)
        return df

    def sample_dataset(self, df: pd.DataFrame, n: int = SAMPLE_SIZE, seed: int = 42) -> pd.DataFrame:
        if n >= len(df):
            return df.reset_index(drop=True)
        frac = n / len(df)
        sampled = (
            df.groupby("label", group_keys=False)
              .apply(lambda g: g.sample(frac=frac, random_state=seed))
        )
        return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)

    def preprocess_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        tokens = word_tokenize(text.lower())
        return " ".join(
            self.lemmatizer.lemmatize(w)
            for w in tokens
            if w.isalpha() and w not in self.stop_words
        )

    def exploratory_data_analysis(self, df: pd.DataFrame, output_dir: str = ".") -> None:
        os.makedirs(output_dir, exist_ok=True)
        labels = df["label"].unique()
        n_cls  = len(labels)

        counts = df["label"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(counts.index.astype(str), counts.values,
                      color=sns.color_palette("muted", len(counts)))
        ax.bar_label(bars, fmt="%d", padding=3)
        ax.set_title("Class Distribution")
        ax.set_xlabel("Label")
        ax.set_ylabel("Count")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/en_1_class_distribution.png")
        plt.show()

        df["text_length"] = df["text"].apply(lambda x: len(x.split()))
        fig, axes = plt.subplots(1, n_cls, figsize=(5 * n_cls, 4))
        if n_cls == 1:
            axes = [axes]
        for ax, lbl in zip(axes, sorted(labels)):
            ax.hist(df[df["label"] == lbl]["text_length"], bins=40,
                    color=sns.color_palette("muted")[list(sorted(labels)).index(lbl)],
                    edgecolor="white")
            ax.set_title(f"Text Length – '{lbl}'")
            ax.set_xlabel("Word Count")
            ax.set_ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/en_2_text_length.png")
        plt.show()

        fig, axes = plt.subplots(1, n_cls, figsize=(7 * n_cls, 5))
        if n_cls == 1:
            axes = [axes]
        for ax, lbl in zip(axes, sorted(labels)):
            tokens = [w for w in " ".join(df[df["label"] == lbl]["text"].values).lower().split()
                      if w.isalpha() and w not in self.stop_words and len(w) > 2]
            top20 = Counter(tokens).most_common(20)
            if top20:
                words, freqs = zip(*top20)
                ax.barh(words[::-1], freqs[::-1],
                        color=sns.color_palette("muted")[list(sorted(labels)).index(lbl)])
            ax.set_title(f"Top-20 Words – '{lbl}'")
            ax.set_xlabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/en_3_top_words.png")
        plt.show()

        fig, axes = plt.subplots(1, n_cls, figsize=(7 * n_cls, 4))
        if n_cls == 1:
            axes = [axes]
        for ax, lbl in zip(axes, sorted(labels)):
            wc = WordCloud(width=600, height=350, background_color="white",
                           colormap="Blues", stopwords=self.stop_words,
                           max_words=120, collocations=False
                           ).generate(" ".join(df[df["label"] == lbl]["text"].values))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(f"Word Cloud – '{lbl}'")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/en_4_wordclouds.png")
        plt.show()

    def text_embedding(self, texts: pd.Series, max_features: int = MAX_FEATURES):
        self.tfidf = TfidfVectorizer(
            max_features=max_features, ngram_range=(1, 2), sublinear_tf=True
        )
        X = self.tfidf.fit_transform(texts).toarray()
        print(f"English feature matrix: {X.shape}")
        return X

    def training(self, X: np.ndarray, y: np.ndarray):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.model = LogisticRegression()
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        return self.model, X_test, y_test, y_pred

    def evaluate(self, y_test: np.ndarray, y_pred: np.ndarray) -> None:
        print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred))

        labels = sorted(np.unique(y_test))
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.5),
                                        max(4, len(labels) * 1.3)))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title("Confusion Matrix – English")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        plt.tight_layout()
        plt.savefig("en_5_confusion_matrix.png")
        plt.show()

    def save_model(self, path: str = "english_gnb_model.pkl") -> None:
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "vectorizer": self.tfidf}, f)
        print(f"English model saved → {path}")

    @staticmethod
    def load_model(path: str = "english_gnb_model.pkl"):
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        return bundle["model"], bundle["vectorizer"]

    def predict(self, texts: list, model=None, vectorizer=None):
        mdl = model or self.model
        vec = vectorizer or self.tfidf
        cleaned = [self.preprocess_text(t) for t in texts]
        X = vec.transform(cleaned).toarray()
        return mdl.predict(X)

    def run_full_pipeline(self, output_dir: str = "english_outputs") -> None:
        df = self.get_english_data()
        df = self.sample_dataset(df, n=SAMPLE_SIZE)
        df["clean_text"] = df["text"].apply(self.preprocess_text)
        print(df.columns)
        # self.exploratory_data_analysis(self,df, output_dir=output_dir)

        X = self.text_embedding(df["clean_text"])
        y = df["label"].values

        _, _, y_test, y_pred = self.training(X, y)
        self.evaluate(y_test, y_pred)
        self.save_model()

if __name__ == "__main__":
    clf = EnglishTextClassifier()
    clf.run_full_pipeline()

