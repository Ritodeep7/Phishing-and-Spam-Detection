from flask import Flask,request,jsonify,render_template
import pickle
import Extraction
import numpy as np
import re
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer


clf1 = pickle.load(open('spam_model.pkl','rb'))
clf2 = pickle.load(open('ctb.pkl','rb'))
tfidf = pickle.load(open('vectorizer.pkl','rb'))
nltk.download('punkt')

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET'])
def predict():
    msg = request.form.get('message')
    msg = str(msg)
    transformed_sms = transform_text(msg)
    vector_input = tfidf.transform([transformed_sms])
    spam = clf1.predict(vector_input)[0]
    phish = 0
    try:
        msg = re.search("(?P<url>https?://[^\s]+)",msg ).group("url")
        pred = Extraction.featureExtraction(msg)
        pred = np.array(pred).reshape((1,-1))[0]
        phish = clf2.predict(pred)
    except:
        phish = 0

    if(phish == 1):
        output = "Suspected Phishing Site"
    elif(spam == 1 and phish == 0):
        output = "Suspicous"
    else:
        output = "Safe"
            

    return jsonify({'result':output})

    #return render_template('index.html', prediction_text=output)


if __name__ == '__main__':
    app.run(debug=True)



