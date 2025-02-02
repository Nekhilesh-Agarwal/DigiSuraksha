import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
def classify_crime_report(report):
    sentiment_score = sia.polarity_scores(report)["compound"]
    if sentiment_score < -0.5:
        return "🚨 High Urgency (Immediate Action Required)"
    elif -0.5 <= sentiment_score < 0:
        return "⚠️ Moderate Urgency (Needs Investigation)"
    else:
        return "✅ Low Urgency (Monitor & Review)"
crime_reports = [
    "Help! A man is being attacked on the street right now!",
    "I saw a group of people acting suspiciously near the bank.",
    "Last night, there was some loud noise in my neighborhood.",
    "A wanted terrorist is outside my house"
]
for report in crime_reports:
    urgency = classify_crime_report(report)
    print(f"Report: {report}\nUrgency Level: {urgency}\n")
