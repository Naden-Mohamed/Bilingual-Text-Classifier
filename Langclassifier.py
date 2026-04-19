import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from Arabictext import ArabicTextClassifier
from Englishtext import EnglishTextClassifier


SAMPLE_PER_LANG = 20_000
MAX_FEATURES     = 5_000
MODEL_PATH       = "lang_classifier_model.pkl"


class LanguageClassifier:

    def __init__(self):
        self.model  = LogisticRegression(max_iter=1000)
        self.tfidf  = None
        self._ar_clf = ArabicTextClassifier()
        self._en_clf = EnglishTextClassifier()

    def build_language_dataset(self) -> pd.DataFrame:
        print("Loading Arabic data …")
        ar_raw = self._ar_clf.get_arabic_data()
        ar_raw.columns = [c.strip().lower() for c in ar_raw.columns]
        ar_df  = self._ar_clf.sample_dataset(ar_raw, n=SAMPLE_PER_LANG)
        ar_df  = self._ar_clf.full_preprocess(ar_df)
        ar_df  = ar_df[["clean_text"]].rename(columns={"clean_text": "text"})
        ar_df["language"] = "Arabic"

        print("Loading English data …")
        en_raw = self._en_clf.get_english_data()
        en_df  = self._en_clf.sample_dataset(en_raw, n=SAMPLE_PER_LANG)
        en_df["clean_text"] = en_df["text"].apply(self._en_clf.preprocess_text)
        en_df  = en_df[["clean_text"]].rename(columns={"clean_text": "text"})
        en_df["language"] = "English"

        combined = (
            pd.concat([ar_df, en_df], axis=0, ignore_index=True)
              .sample(frac=1, random_state=42)
              .reset_index(drop=True)
        )
        print(f"Combined dataset: {len(combined):,} rows")
        print(combined["language"].value_counts().to_string())
        return combined

    def text_embedding(self, texts: pd.Series, fit: bool = True):
        if fit:
            self.tfidf = TfidfVectorizer(
                max_features=MAX_FEATURES, ngram_range=(1, 2),
                sublinear_tf=True, analyzer="char_wb"
            )
            X = self.tfidf.fit_transform(texts).toarray()
        else:
            X = self.tfidf.transform(texts).toarray()
        print(f"Language classifier feature matrix: {X.shape}")
        return X

    def training(self, X: np.ndarray, y: np.ndarray):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print("Training language classifier …")
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        return X_test, y_test, y_pred

    def evaluate(self, y_test: np.ndarray, y_pred: np.ndarray) -> None:
        print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred))

        labels = sorted(np.unique(y_test))
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title("Confusion Matrix – Language Classifier")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.tight_layout()
        plt.savefig("lang_confusion_matrix.png")
        plt.show()

    def save_model(self, path: str = MODEL_PATH) -> None:
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "vectorizer": self.tfidf}, f)
        print(f"Language classifier saved → {path}")

    @staticmethod
    def load_model(path: str = MODEL_PATH):
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        return bundle["model"], bundle["vectorizer"]

    def detect_lang(self, text: str) -> str:
        X = self.tfidf.transform([text]).toarray()
        return self.model.predict(X)[0]
    def run_full_pipeline(self) -> None:
        df = self.build_language_dataset()
        X  = self.text_embedding(df["text"], fit=True)
        y  = df["language"].values
        _, y_test, y_pred = self.training(X, y)
        self.evaluate(y_test, y_pred)
        self.save_model()