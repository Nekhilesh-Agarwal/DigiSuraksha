from flask import Flask, request, jsonify
import joblib

# Load the pre-trained model and vectorizer using relative paths (as Render will automatically manage files)
model = joblib.load('crime_report_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

app = Flask(__name__)

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

# No need to run the app manually when hosting on Render.
# If running locally, you can uncomment the below line.
# if __name__ == '__main__':
#     app.run(debug=True)
