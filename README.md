# 🧠 Mood-Miner

Mood-Miner is an all-in-one Streamlit web app for sentiment analysis using both a custom-trained Naive Bayes machine learning model and TextBlob. It lets you analyze sentiment for single sentences or process entire CSV files in bulk.

---

## 🚀 Features

- Analyze sentiment for single sentences instantly
- Batch CSV upload (column: `Text`) with downloadable results
- Dual sentiment analysis: Machine Learning (Naive Bayes) & TextBlob
- Clean, modern Streamlit UI
- No separate training script: model is trained, saved, and used within the same app file

---

## 🛠️ Tech Stack

- **Python 3.x**
- **Streamlit** 
- **Scikit-learn**
- **Pandas**
- **TextBlob**
- **NLTK**

---

## 📦 How to Run

1. **Place your app file and dataset together**

   - Make sure you have your **`app.py`** and a training CSV file (e.g. `amazon_reviews.csv`) with columns: `reviewText` and `rating`.

2. **Install dependencies:**

    pip install -r requirements.txt


3. **Start the app:**

    streamlit run app.py


---

## 🌟 Example Training CSV Format

<pre> ### 📄 Example Training CSV Format You can prepare a CSV file like this: | reviewText | rating | |------------------------------------------|--------| | This product was excellent. | 5 | | Bad experience, would not buy again. | 1 | | Average, nothing special. | 3 | </pre>

## 🌐 Usage

- Enter a sentence in the left panel for instant analysis, or
- Upload a CSV with a `Text` column for batch analysis (results and download link provided).



