import streamlit as st
from transformers import pipeline

pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")




st.title("Hello, Streamlit!")


text = st.text_input("Enter some text to analyze its sentiment:")

if text:
    result = pipe(text)
    st.write("Sentiment Analysis Result:")
    st.write(result[0]['label'])
    st.write(f"Confidence: {result[0]['score']:.2f}")