import streamlit as st
import pandas as pd
import pickle
from textblob import TextBlob

# Import your text cleaner (must be safe to import!)
from train_model import clean_text

# Load model and vectorizer
with open('trained_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# ------ Streamlit UI ------
st.set_page_config(page_title="Sentiment Analysis App", layout="centered")
st.title("🧠 Mood-Miner")
st.write("Enter text or upload a CSV to analyze sentiment using TextBlob and Naive Bayes.")

# ---- Single Text Input ----
text_input = st.text_area("Enter a sentence")

if st.button("Analyze Text"):
    if text_input:
        blob = TextBlob(text_input)
        polarity = blob.sentiment.polarity
        sentiment_label = (
            "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
        )

        # Preprocess input before prediction!
        cleaned_input = clean_text(text_input)
        X_vec = vectorizer.transform([cleaned_input])
        model_pred = model.predict(X_vec)[0]

        st.write(f"**Polarity Score**: `{polarity:.2f}`")
        st.write(f"**TextBlob Sentiment**: {sentiment_label}")
        st.success(f"**Naive Bayes Prediction**: `{model_pred}`")
    else:
        st.warning("Please enter some text.")

# ---- File Upload for Bulk Sentiment Analysis ----
st.markdown("---")
st.markdown("### 📁 Upload CSV for Bulk Sentiment Analysis")

file = st.file_uploader("Upload a CSV file with a `Text` column", type="csv")

if file is not None:
    try:
        df = pd.read_csv(file)
        # Ensure column exists (case-insensitive)
        text_col = None
        for col in df.columns:
            if col.strip().lower() == 'text':
                text_col = col
                break
        if text_col:
            df[text_col] = df[text_col].astype(str)
            df['Polarity'] = df[text_col].apply(lambda x: TextBlob(x).sentiment.polarity)
            df['TextBlob_Sentiment'] = df['Polarity'].apply(
                lambda p: 'Positive' if p > 0 else 'Negative' if p < 0 else 'Neutral'
            )
            # Preprocess all input before prediction!
            df['Cleaned_Text'] = df[text_col].apply(clean_text)
            X_vec = vectorizer.transform(df['Cleaned_Text'])
            df['Model_Prediction'] = model.predict(X_vec)

            st.write("### 🧾 Results Sample")
            st.dataframe(df[[text_col, 'Polarity', 'TextBlob_Sentiment', 'Model_Prediction']].head())
            st.download_button("📥 Download Full Results", df.to_csv(index=False), file_name="results.csv")
        else:
            st.error("CSV must contain a `Text` column (case-insensitive).")
    except Exception as e:
        st.error(f"Error processing file: {e}")
