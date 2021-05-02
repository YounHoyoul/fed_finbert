import os
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
import string
import random

import pickle
import codecs
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import ensemble, metrics, model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from xgboost.sklearn import XGBClassifier

from lime import lime_text
from lime.lime_text import LimeTextExplainer

from bs4 import BeautifulSoup
import boto3

class ExplainPredict:
    def __init__(self, xgb_path, tfidf_path, temp_path):
        self.loaded_xgb = pickle.load(open(xgb_path, "rb"))
        self.tfidf = pickle.load(open(tfidf_path, "rb"))
        self.temp_path = temp_path

    def clean_sentence(self, sentence):
        stopword_add_list = ['pdf','presentation','slide','slides','q','myplayer','download','â','participant','and','the','of']

        sentence = re.sub(r'\([^)]*\)', '', sentence.lower()) # Remove Sound-effect labels
        sentence = sentence.encode('ascii', errors='ignore').strip().decode('ascii')

        tokenized_doc = word_tokenize(sentence)
        tokenized_doc = [w for w in tokenized_doc if not w in stopword_add_list]#[:1024]

        return ' '.join(tokenized_doc)

    def get_random_string(self, length):
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for i in range(length))

    def predict(self, s_doc, is_upload=True):
        class_names = ['lower', 'maintain', 'raise']
        c_tf = make_pipeline(self.tfidf, self.loaded_xgb)
        explainer_tf = LimeTextExplainer(class_names=class_names)
        exp = explainer_tf.explain_instance(self.clean_sentence(s_doc), c_tf.predict_proba, num_features=10, top_labels=1)
        
        # generate a file name
        filename = self.get_random_string(10) + '.html'
        lime = exp.save_to_file(self.temp_path + filename)

        if is_upload :
            # upload a file
            self.upload(filename)

        # return filename
        return filename

        # lime = exp.save_to_file("./output/explainer.html")
        # f = codecs.open("./output/explainer.html", "r", "utf-8")
        # return f.read()

    def upload(self, filename):
        s3 = boto3.resource("s3")

        with open(self.temp_path + filename) as inf:
            data = inf.read()
            soup = BeautifulSoup(data, features="html.parser")

        soup.head.script.decompose()

        soup.head.append(BeautifulSoup("<script src=\"https://finviznlp-uploads.s3.us-east-2.amazonaws.com/static/d3_lib.js\"></script>", features="html.parser"))
        soup.head.append(BeautifulSoup("<script src=\"https://finviznlp-uploads.s3.us-east-2.amazonaws.com/static/iframe.js\"></script>", features="html.parser"))
        soup.head.append(BeautifulSoup("<link href=\"https://finviznlp-uploads.s3.us-east-2.amazonaws.com/static/iframe.css\" rel=\"stylesheet\"/>", features="html.parser"))

        tmp = self.temp_path+ filename+".pp"

        with open(tmp, "w") as outf:
            outf.write(str(soup))

        s3.meta.client.upload_file(tmp, "finviznlp-uploads", filename, ExtraArgs={'ContentType': 'text/html'})