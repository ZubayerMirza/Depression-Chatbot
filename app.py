import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['POST'])
def chat():
    input_data = request.get_json()
    message = input_data['message']
    processed_message = preprocess(message)
    response = get_response(processed_message)
    return jsonify({'message': response})

def preprocess(message):
    porter_stemmer = PorterStemmer()
    stopwords_set = set(stopwords.words('english'))
    message = message.lower()
    words = nltk.word_tokenize(message)
    words = [porter_stemmer.stem(word) for word in words if word not in stopwords_set]
    return ' '.join(words)

def get_response(message):
    model = MultinomialNB()
    vectorizer = CountVectorizer()
    data = pd.read_csv('depression_data.csv')
    X = vectorizer.fit_transform(data['message']).toarray()
    y = data['label'].values
    model.fit(X, y)
    processed_message = vectorizer.transform([message]).toarray()
    prediction = model.predict(processed_message)
    if prediction == 0:
        return 'I am sorry to hear that. How can I help you?'
    else:
        return 'Please seek help from a professional. Here are some resources: https://www.nimh.nih.gov/health/find-help/index.shtml'

if __name__ == '__main__':
    app.run()
