
# Bilingual Text Classifier (Arabic  + English )

>  An end-to-end NLP system that detects language and classifies sentiment for both Arabic and English text — built with production-ready structure, optimized preprocessing, and an interactive UI.

Try it: https://f42309da46a7503e25.gradio.live/ 

##  Key Highlights

✨ Built a **dual-language NLP pipeline** (Arabic + English)
✨ Designed **custom Arabic preprocessing** (normalization, diacritics removal, reshaping)
✨ Implemented **TF-IDF + Logistic Regression** for efficient classification
✨ Created **EDA visualizations & word clouds (Arabic-compatible)**
✨ Developed an **interactive Gradio UI** for real-time predictions
✨ Structured the project for **scalability & reuse**

---

##  How It Works

```id="flow123"
User Input → Language Detection → Text Preprocessing → TF-IDF → Model Prediction → UI Output
```

###  Pipeline Breakdown

1. **Language Detection**

   * Automatically identifies Arabic vs English

2. **Text Preprocessing**

   * Arabic-specific normalization (أ → ا, etc.)
   * Stopword removal (NLTK)
   * Noise removal (URLs, emojis, punctuation)

3. **Feature Engineering**

   * TF-IDF (unigrams + bigrams)

4. **Model**

   * Logistic Regression (fast, interpretable, scalable)

5. **Output**

   * Sentiment classification via Gradio UI

---

## 📊 Visual Insights

The project includes rich exploratory analysis:

*  Class distribution
*  Text length analysis
*  Top frequent words
*  **Arabic word clouds (properly reshaped & RTL-correct)**

---

## 🖥️ Demo

### Example Input:

```id="demo001"
الخدمة كانت ممتازة وسريعة
```

### Output:

```id="demo002"
Language: Arabic
Sentiment: Positive
```

---

## 🛠️ Tech Stack

* **Python**
* **Scikit-learn**
* **NLTK**
* **Pandas / NumPy**
* **Matplotlib / Seaborn**
* **WordCloud**
* **arabic_reshaper + bidi**
* **Gradio**

---

## Running the Project

```bash id="run001"
git clone https://github.com/Naden-Mohamed/Bilingual-Text-Classifier.git
cd Bilingual-Text-Classifier

python -m venv .venv
source .venv/Scripts/activate

pip install -r requirements.txt
python main.py
```

## Challenges Solved

* Arabic text rendering issues → Fixed using reshaping + bidi
*  Slow word cloud generation → Optimized processing strategy
*  Pipeline crashes (missing columns / NoneType errors) → Added validation + safeguards
