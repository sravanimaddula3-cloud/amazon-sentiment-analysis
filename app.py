import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Title
st.title("Amazon Product Review Sentiment Analysis")
st.write("Machine Learning based Sentiment Analysis using Real CSV Dataset")

# Load Dataset
data = pd.read_csv("amazon_reviews.csv")
data["sentiment"] = data["overall"].apply(lambda x: "positive" if x >= 4 else "negative")

# Remove missing values
data = data.dropna(subset=["reviewText", "sentiment"])

# Display Dataset Preview
st.subheader("Dataset Preview")
st.dataframe(data.head())

# Features and Labels
X = data["reviewText"]
y = data["sentiment"]

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words="english")

X_tfidf = tfidf.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42
)

# Logistic Regression Model
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

st.write(f"### Model Accuracy: {round(accuracy * 100, 2)}%")

# Classification Report
st.subheader("Classification Report")

report = classification_report(y_test, y_pred)

st.text(report)

# User Input
st.subheader("Sentiment Prediction")

user_review = st.text_area("Enter your product review")

if st.button("Predict Sentiment"):

    if user_review:

        review_data = tfidf.transform([user_review])

        prediction = model.predict(review_data)

        confidence = model.predict_proba(review_data)

        st.success(
            f"Predicted Sentiment: {prediction[0]}"
        )

        st.write(
            f"Confidence Score: {round(max(confidence[0]) * 100, 2)}%"
        )

    else:

        st.warning("Please enter a review first")