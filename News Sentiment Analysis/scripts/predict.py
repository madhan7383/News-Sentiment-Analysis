import joblib
from scripts.preprocess import preprocess_text

# Load the model and vectorizer
model = joblib.load('models/sentiment_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

def predict_sentiment(text):
    """Predict the sentiment of a given text (positive, negative, or neutral)."""
    processed_text = preprocess_text(text)
    text_tfidf = vectorizer.transform([processed_text])
    prediction = model.predict(text_tfidf)
    return prediction[0]

if __name__ == "__main__":
    sample_text = "This is a great day in the world of technology."
    sentiment = predict_sentiment(sample_text)
    print(f"Sentiment: {sentiment}")
