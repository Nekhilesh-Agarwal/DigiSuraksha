from flask import Flask, request, jsonify
import joblib
import spacy
from textblob import TextBlob

# Load the pre-trained model and vectorizer
model = joblib.load('crime_report_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Define common vague words and non-crime-related words
VAGUE_WORDS = {
    "someone", "somewhere", "something", "bad thing", "strange", "happened", "incident", "event",
    "thing", "stuff", "weird", "unclear", "unknown", "odd", "unusual", "unsure", "maybe", "possibly",
    "probably", "kind of", "sort of", "felt like", "seemed", "apparently", "random", "mysterious",
    "unidentified", "vague", "confusing", "indistinct", "blurred", "unconfirmed", "iffy", "unverified"
}


app = Flask(__name__)

def detect_unusual_language(text):
    """
    Detects unusual language patterns using NLP.
    """
    doc = nlp(text)
    
    # Detect vague terms
    vague_terms = [token.text.lower() for token in doc if token.text.lower() in VAGUE_WORDS]
    
    # Detect excessive adjectives and adverbs (over-dramatization)
    adjectives = [token.text for token in doc if token.pos_ in ["ADJ", "ADV"]]
    if len(adjectives) > 4:
        vague_terms.append(f"Too many adjectives/adverbs: {', '.join(adjectives)}")

    # Use TextBlob for sentiment analysis
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    if sentiment_score > 0.7:  # Overly positive or dramatic language
        vague_terms.append(f"Overly dramatic sentiment detected (Score: {sentiment_score})")

    # Check if report lacks specific entities (like person, location)
    entities = [ent.label_ for ent in doc.ents]
    if "GPE" not in entities and "PERSON" not in entities:
        vague_terms.append("No specific person or location mentioned.")

    return bool(vague_terms), vague_terms

@app.route('/')
def home():
    return "Crime Report Urgency Prediction API"

@app.route('/predict_urgency', methods=['POST'])
def predict_urgency():
    data = request.get_json()
    crime_report = data.get('crime_report', '')   

    if not crime_report:
        return jsonify({"error": "Crime report is required"}), 400

    # Transform the input crime report text
    crime_report_transformed = vectorizer.transform([crime_report])

    # Predict urgency using the model
    urgency = model.predict(crime_report_transformed)[0]

    return jsonify({"urgency": urgency})

@app.route('/predict_authenticity', methods=['POST'])
def predict_authenticity():
    data = request.get_json()
    crime_report = data.get('crime_report', '')

    if not crime_report:
        return jsonify({"error": "Crime report is required"}), 400

    # Run NLP-based authenticity checks
    flagged, reasons = detect_unusual_language(crime_report)

    response = {"authentic": not flagged}
    
    if flagged:
        response["flagged"] = True
        response["reasons"] = reasons
    
    return jsonify(response)