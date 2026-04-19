import gradio as gr
from Langclassifier import LanguageClassifier
from Arabictext import ArabicTextClassifier
from Englishtext import EnglishTextClassifier
global models_loaded
import os

lang_clf = LanguageClassifier()
ar_clf   = ArabicTextClassifier()
en_clf   = EnglishTextClassifier()

models_loaded = False

def load_models():

    if not os.path.exists("lang_classifier_model.pkl"):
        lang_clf.run_full_pipeline()

    if not os.path.exists("arabic_gnb_model.pkl"):
        ar_clf.run_full_pipeline()

    if not os.path.exists("english_gnb_model.pkl"):
        en_clf.run_full_pipeline()

    lang_clf.model, lang_clf.tfidf = LanguageClassifier.load_model()
    ar_clf.model,   ar_clf.tfidf   = ArabicTextClassifier.load_model()
    en_clf.model,   en_clf.tfidf   = EnglishTextClassifier.load_model()

    models_loaded = True
    
    
def classify(text: str) -> str:
    global models_loaded

    if not models_loaded:
        load_models()

    if not text or not text.strip():
        return "Please enter some text."

    text = text.strip()
    language = lang_clf.detect_lang(text)

    if language == "Arabic":
        prediction = ar_clf.predict([text])[0]
    else:
        prediction = en_clf.predict([text])[0]

    return f"Language : {language}\nCategory : {prediction}"

demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=4, placeholder="Enter Arabic or English text here …", label="Input Text"),
    outputs=gr.Textbox(label="Result"),
    title="Arabic / English Text Classifier",
    description="Detects language then classifies sentiment/category using trained Logistic Regression models.",
    api_name="classify"
)

if __name__ == "__main__":
    load_models()
    demo.launch()