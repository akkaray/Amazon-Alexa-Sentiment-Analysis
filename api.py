import warnings
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd
import pickle

# Suppress warnings related to sklearn model loading
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

STOPWORDS = set(stopwords.words("english"))

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/test", methods=["GET"])
def test():
    return "Test request received successfully. Service is running."

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("landing.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Select the predictor to be loaded from Models folder
    predictor = pickle.load(open(r"Models/best_model_xgb.pkl", "rb"))
    scaler = pickle.load(open(r"Models/scaler.pkl", "rb"))  # Load scaler if necessary
    tfidf_vectorizer = pickle.load(open(r"Models/tfidfVectorizer.pkl", "rb"))
    
    try:
        if "file" in request.files:
            # Handle file uploads for bulk predictions (removed)
            return jsonify({"error": "Bulk prediction functionality has been removed."})
        
        elif "text" in request.json:
            # Single string prediction
            text_input = request.json["text"]
            predicted_sentiment = single_prediction(predictor, scaler, tfidf_vectorizer, text_input)
            return jsonify({"prediction": predicted_sentiment})

    except Exception as e:
        return jsonify({"error": str(e)})

def single_prediction(predictor, scaler, tfidf_vectorizer, text_input):
    corpus = []
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
    review = " ".join(review)
    corpus.append(review)
    X_prediction = tfidf_vectorizer.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)  # If using a scaler
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)[0]

    return "Positive" if y_predictions == 1 else "Negative"

if __name__ == "__main__":
    app.run(port=5000, debug=True)
