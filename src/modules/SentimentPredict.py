import pysentiment2 as ps
from textblob import TextBlob

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

class SentimentPredict:
    def __init__(self):
        self.lm = ps.LM()

    def predict(self, s_doc):
        finance = self.lm.get_score(self.lm.tokenize(s_doc))
        general = TextBlob(s_doc).sentiment

        return {
            "word_count" : len(str(s_doc).split()),
            "sentence_count" : len(sent_tokenize(s_doc)), 
            "polarityGeneral": general.polarity,
            "subjectivityGeneral": general.subjectivity,
            "polarityFinance": finance['Polarity'],
            "subjectivityFinance": finance['Subjectivity']
        }
