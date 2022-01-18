import pickle
import Extraction
import re
import numpy as np
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

#load the pickle file
clf1 = pickle.load(open('spam_model.pkl','rb'))
clf2 = pickle.load(open('rf.pkl','rb'))
tfidf = pickle.load(open('vectorizer.pkl','rb'))


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



#input url
print("enter msg")
str = input()
transformed_sms = transform_text(str)
vector_input = tfidf.transform([transformed_sms])
result = clf1.predict(vector_input)[0]
if result == 1:
        str = re.search("(?P<url>https?://[^\s]+)",str ).group("url")
        if(len(str)!=0):
            pred = Extraction.featureExtraction(str)
            pred = np.array(pred).reshape((1,-1))
            prediction = clf2.predict(pred)
            print(prediction)
print(result)

