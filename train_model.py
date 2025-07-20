# Import necessary libraries
import pandas as pd                     # Data handling
import re                               # Regular expressions for text cleaning
import nltk                             # Natural Language Toolkit
from nltk.corpus import stopwords       # Stopwords removal
from sklearn.model_selection import train_test_split  # Splitting dataset
from sklearn.feature_extraction.text import CountVectorizer  # Convert text into features
from sklearn.naive_bayes import MultinomialNB         # Naive Bayes classifier
from sklearn.metrics import classification_report, accuracy_score  # Performance metrics
import pickle                           # Save model and vectorizer

# Download stopwords if not already present
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# -------------------------------------
# Function to clean review text
# -------------------------------------
def clean_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation, digits, keep alphabets only
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# -------------------------------------
# Function to convert numeric star ratings to sentiment labels
# -------------------------------------
def label_sentiment(star):
    # star needs to be numeric
    try:
        star = float(star)
    except:
        return "Unknown"
    if star >= 4:
        return "Positive"
    elif star == 3:
        return "Neutral"
    else:
        return "Negative"

# -------------------------------------
# Main training pipeline
# -------------------------------------
def main():
    # 📄 Path to your CSV dataset
    csv_file_path = "amazon_reviews.csv"   # <-- Change if your file has a different name

    # 📥 Load CSV file into DataFrame
    df = pd.read_csv(csv_file_path)

    # 🔍 Check if required columns exist
    if 'reviewText' not in df.columns or 'rating' not in df.columns:
        raise ValueError("CSV file must contain 'reviewText' and 'rating' columns.")

    # 🚿 Drop rows with missing review or rating values
    df = df[['reviewText', 'rating']].dropna()

    # 🧽 Clean review text using the clean_text function
    df['cleaned_text'] = df['reviewText'].apply(clean_text)

    # 🏷️ Label sentiments from star ratings
    df['Sentiment'] = df['rating'].apply(label_sentiment)

    # 🎯 Features and Labels
    X = df['cleaned_text']
    y = df['Sentiment']

    # ✂️ Split dataset into training and test sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # 🔠 Convert text data to numerical vectors using bag-of-words
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 🧠 Train Naive Bayes model for text classification
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # 📊 Predict and Evaluate
    y_pred = model.predict(X_test_vec)
    print("\n✅ Model Accuracy:", accuracy_score(y_test, y_pred))
    print("\n📋 Classification Report:\n")
    print(classification_report(y_test, y_pred))

    # 💾 Save trained model and vectorizer for later use
    with open('trained_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    with open('vectorizer.pkl', 'wb') as vec_file:
        pickle.dump(vectorizer, vec_file)

    print("✅ Model and vectorizer saved to disk successfully.")

# ---------------------------------------------------------------------
# Ensure training runs only when called directly (not on import)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()









