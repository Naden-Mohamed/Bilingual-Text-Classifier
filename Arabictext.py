import os
import re
import io
import pickle
import zipfile

import emoji
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
import pyarabic.araby as araby
from arabic_reshaper import reshape
from bidi.algorithm import get_display
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score
)

load_dotenv()
nltk.download("stopwords", quiet=True)

USER = os.getenv("KAGGLE_USERNAME")
KEY  = os.getenv("KAGGLE_API_TOKEN")

ARABIC_STOPWORDS = set(stopwords.words("arabic"))
SAMPLE_SIZE  = 60_000
MAX_FEATURES = 3_000


def _ar(text: str) -> str:
    return get_display(reshape(str(text)))


class ArabicTextClassifier:

    def __init__(self):
        self.tfidf = None
        self.model = None

    def get_arabic_data(self) -> pd.DataFrame:
        url = "https://www.kaggle.com/api/v1/datasets/download/abedkhooli/arabic-100k-reviews"
        response = requests.get(url, auth=HTTPBasicAuth(USER, KEY))
        if response.status_code != 200:
            raise Exception(f"Download failed – HTTP {response.status_code}")
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            file_name = zf.namelist()[0]
            df = pd.read_csv(zf.open(file_name), sep="\t", encoding="utf-8-sig")
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.rename(columns={"sentiment": "label", "review": "text"})
        return df

    def sample_dataset(self, df: pd.DataFrame, n: int = SAMPLE_SIZE, seed: int = 42) -> pd.DataFrame:
        df = df.copy()

        if "label" not in df.columns:
            raise ValueError("Label column missing before sampling!")

        df = df.dropna(subset=["label"])

        if n >= len(df):
            return df.reset_index(drop=True)

        return df.sample(n=n, random_state=seed).reset_index(drop=True)
    def normalizeArabic(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.strip()
        text = re.sub("ى", "ي", text)
        text = re.sub("ؤ", "ء", text)
        text = re.sub("ئ", "ء", text)
        text = re.sub("ة", "ه", text)
        text = re.sub("[إأٱآا]", "ا", text)
        text = text.replace("وو", "و").replace("يي", "ي").replace("اا", "ا")
        text = re.sub(r"(.)\1+", r"\1\1", text)
        text = araby.strip_tashkeel(text)
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
        text = re.sub(r"http\S+|www\.\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = emoji.replace_emoji(text, replace="")
        return text.strip()

    def remove_stopwords(self, text: str) -> str:
        return " ".join(w for w in str(text).split() if w not in ARABIC_STOPWORDS)

    def remove_extra_spaces(self, text: str) -> str:
        return " ".join(text.split())

    def full_preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "label" not in df.columns:
            raise ValueError("Label lost before preprocessing!")

        df["clean_text"] = (
            df["text"]
            .apply(self.normalizeArabic)
            .apply(self.remove_stopwords)
            .apply(self.remove_extra_spaces)
        )

        df = df[df["clean_text"].str.strip().astype(bool)].reset_index(drop=True)
        return df

    def exploratory_data_analysis(self, df: pd.DataFrame, output_dir: str = ".") -> None:
        os.makedirs(output_dir, exist_ok=True)
        if "label" not in df.columns:
             raise ValueError(f"'label' column missing! Available columns: {df.columns}")
        labels = df["label"].unique()

        counts = df["label"].value_counts()
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(counts.index.astype(str), counts.values,
                      color=["#2196F3", "#F44336", "#4CAF50", "#FF9800"][:len(counts)],
                      edgecolor="white")
        ax.bar_label(bars, fmt="%d", padding=4)
        ax.set_title("Class Distribution")
        ax.set_xlabel("Label")
        ax.set_ylabel("Count")
        ax.set_xticklabels([_ar(str(l)) for l in counts.index])
        plt.tight_layout()
        plt.savefig(f"{output_dir}/ar_1_class_distribution.png", dpi=150)
        plt.show()

        df["text_len"] = df["clean_text"].apply(lambda x: len(x.split()))
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist(df["text_len"], bins=50, color="#5C6BC0", edgecolor="white")
        axes[0].set_title("Overall Text Length")
        axes[0].set_xlabel("Word Count")
        axes[0].set_ylabel("Frequency")
        for lbl, grp in df.groupby("label"):
            axes[1].hist(grp["text_len"], bins=40, alpha=0.6, label=str(lbl))
        axes[1].set_title("Text Length per Class")
        axes[1].set_xlabel("Word Count")
        axes[1].legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/ar_2_text_length.png", dpi=150)
        plt.show()

        n_cls = len(labels)
        n_cols = min(n_cls, 3)
        n_rows = (n_cls + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
        axes = np.array(axes).flatten() if (n_rows * n_cols) > 1 else [axes]
        for ax, lbl in zip(axes, labels):
            tokens = " ".join(df[df["label"] == lbl]["clean_text"]).split()
            top20 = Counter(tokens).most_common(20)
            words = [_ar(w) for w, _ in top20]
            freqs = [f for _, f in top20]
            ax.barh(words[::-1], freqs[::-1], color="#26A69A")
            ax.set_title(_ar(f"Top Words – {lbl}"))
            ax.set_xlabel("Frequency")
        for ax in axes[n_cls:]:
            ax.set_visible(False)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/ar_3_top_words.png", dpi=150, bbox_inches="tight")
        plt.show()

        for lbl in labels:
            print(f"Generating wordcloud for label: {lbl}...")

            subset = df[df["label"] == lbl]["clean_text"].sample(n=5000, random_state=42, replace=True)

            text_blob = " ".join(subset)

            reshaped_text = reshape(text_blob)
            bidi_text = get_display(reshaped_text)

            wc = WordCloud(
                font_path="C:/Windows/Fonts/arial.ttf",
                background_color="white",
                width=1000,
                height=500,
                max_words=150,
                collocations=False
            ).generate(bidi_text)

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(_ar(f"Word Cloud – {lbl}"))

            plt.tight_layout()
            plt.savefig(f"{output_dir}/ar_4_wordcloud_{str(lbl).replace(' ', '_')}.png", dpi=150)
            plt.show()

    def text_embedding(self, texts: pd.Series, max_features: int = MAX_FEATURES):
        self.tfidf = TfidfVectorizer(
            max_features=max_features, ngram_range=(1, 2),
            sublinear_tf=True, min_df=2
        )
        X = self.tfidf.fit_transform(texts).toarray()
        print(f"Arabic feature matrix: {X.shape}")
        return X

    def training(self, X: np.ndarray, y: np.ndarray):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.model = LogisticRegression()
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        return self.model, X_test, y_test, y_pred

    def model_evaluation(self, y_test, y_pred, class_names=None, output_dir="."):
        print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

        os.makedirs(output_dir, exist_ok=True)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(
            ax=ax, colorbar=True, cmap="Blues"
        )
        ax.set_title("Confusion Matrix – Arabic")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/ar_confusion_matrix.png", dpi=150)
        plt.show()

    def predict(self, texts: list, model=None, vectorizer=None):
        mdl = model or self.model
        vec = vectorizer or self.tfidf
        cleaned = [
            self.remove_extra_spaces(self.remove_stopwords(self.normalizeArabic(t)))
            for t in texts
        ]
        X = vec.transform(cleaned).toarray()
        return mdl.predict(X)


    def save_model(self, output_dir=".", filename: str = "arabic_model.pkl") -> None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        path = output_dir + "/" +filename
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "vectorizer": self.tfidf}, f)
        print(f"Arabic model saved → {path}")

    @staticmethod
    def load_model(path: str = "models/arabic_model.pkl"):
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        return bundle["model"], bundle["vectorizer"]


    def run_full_pipeline(self, output_dir: str = "arabic_outputs") -> None:
        df = self.get_arabic_data()
        df = self.sample_dataset(df, n=SAMPLE_SIZE)
        df = self.full_preprocess(df)

        self.exploratory_data_analysis(df, output_dir=output_dir)

        X = self.text_embedding(df["clean_text"])
        y = df["label"].values
        class_names = sorted(df["label"].unique().tolist())

        _, _, y_test, y_pred = self.training(X, y)
        self.model_evaluation(y_test, y_pred, class_names=class_names, output_dir=output_dir)
        self.save_model( output_dir=output_dir)

if __name__ == "__main__":
    clf = ArabicTextClassifier()
    clf.run_full_pipeline()
