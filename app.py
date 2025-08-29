import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt_tab')
nltk.download('stopwords')
tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

# creating a function which perform preprocessing
ps=PorterStemmer()
def transform_text(text):
    text=text.lower() #lower case
    text=nltk.word_tokenize(text) #list

    y=[]
    for i in text:
        if i.isalnum():
            y.append(i) #removing specail charatcer
    text=y[:] #cloning list
    y.clear()
    for i in text: #removing punctuaton and stop words
        if i  not in stopwords.words('english') and  i not in string.punctuation: 
            y.append(i)
    text=y[:]
    y.clear()
    #stemming
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)


st.title("SMS Spam classifier")
input_sms=st.text_area("Enter the message")
if st.button('Predict'):

    #1. preprocess
    transform_sms=transform_text(input_sms)
    #2 vectorize
    vector_input=tfidf.transform([transform_sms])
    #3 predict
    result=model.predict(vector_input)[0]
    #4 display
    if result==1:
        st.header("Spam")
    else:
        st.header("Not Spam")