import numpy as np
from flask import Flask, request, jsonify, render_template
import keras.models
import nltk #Natural Language tool kit
from nltk.corpus import stopwords

from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

import re
import sys 
import os
sys.path.append(os.path.abspath("./model"))

from load import * 
app = Flask(__name__)

global model
model = init()
	

@app.route('/',methods=["GET"])
def index():
	return "200,OK"

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    voc_size=5000

    data = request.get_json()
    messages=data['title']

    from nltk.stem.porter import PorterStemmer
    ps = PorterStemmer()
    review = re.sub('[^a-zA-Z]', ' ', messages)
    review = review.lower()
    review = review.split()
      
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)

    onehot_repr=[one_hot(review,voc_size)] 

    sent_length=50
    embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
    prediction = model.predict([np.array(embedded_docs)])

    output = prediction[0]
    print(output)
    if output>0.5:
        response={
            "status": "Fake"
        }
    else:
        response={
            "status": "Real"
        }
    
    return response
	

if __name__ == "__main__":
    app.run(debug=True)