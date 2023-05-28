import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import class_weight
from PIL import Image
import base64


def add_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url('data:image/png;base64,{encoded_string}');
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


def main():
    add_bg_from_local('#STOPCYBERBULLYING.png')

    st.title("Cyberbullying Detection")
    st.subheader("Enter a tweet to classify if it is offensive or non-offensive")

    # Load the dataset
    df = pd.read_csv('public_data_labeled.csv')

    # Define the set of stopwords
    stop_words = {
        "a", "an", "the", "and", "or", "but", "if", "is", "it", "of", "on", "in", "to", "by", "for"
        # Add more stopwords as needed
    }

    # Create a TF-IDF vectorizer with custom stopwords
    vectorizer = TfidfVectorizer(stop_words=stop_words)

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Transform the data
    X_vectorized = vectorizer.fit_transform(df["full_text"])
    y = df["label"]

    # Calculate class weights
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)

    # Create a naive Bayes classifier with class weights
    clf = MultinomialNB(class_prior=class_weights)

    # Train the classifier
    clf.fit(X_vectorized, y)

    # Get user input
    tweet = st.text_input("Enter a tweet:")
    if tweet:
        # Vectorize the tweet
        tweet_vectorized = vectorizer.transform([tweet])

        # Predict the label of the tweet
        label = clf.predict(tweet_vectorized)[0]

        # Print the prediction
        if label == "Non-offensive":
            st.success("The tweet is classified as Non-offensive.")
            image=Image.open('1(1).png')
            st.image(image)
        else:
            st.error("The tweet is classified as Offensive.")
            image = Image.open('2.png')
            st.image(image)


if __name__ == '__main__':
    main()
