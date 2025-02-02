import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
df = pd.read_csv('dataset.csv')
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['Crime Report'])
y = df['Urgency Label']
model = LogisticRegression(max_iter=1000)
model.fit(X, y)
joblib.dump(model, 'crime_report_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("Model and vectorizer saved successfully!")
