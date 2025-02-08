import streamlit as st
from transformers import pipeline
from textblob import TextBlob

pipe = pipeline('sentiment-analysis')
st.title("Sentiment Analysis")

st.subheader("Which framework would you like to use for sentiment analysis?")
option = st.selectbox('Choose Framework:',('Transformers', 'TextBlob')) # option is stored in this variable

st.subheader("Enter the text you want to analyze")
text = st.text_input('Enter text:') # text is stored in this variable

if option == 'Transformers':
    out = pipe(text)
else:
    out = TextBlob(text)
    out = out.sentiment
    
st.write("Sentiment of Text: ")
st.write(out)