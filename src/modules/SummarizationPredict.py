from transformers import pipeline 
from summa.summarizer import summarize
import nltk
import re
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('punkt')

class SummarizationPredict:
    def __init__(self):
        self.summarizer = pipeline('summarization')

    def clean_sentence(self, sentence):
        stopword_add_list = ['pdf','presentation','slide','slides','q','myplayer','download','â','participant']

        sentence = re.sub(r'\([^)]*\)', '', sentence) # Remove Sound-effect labels
        sentence = sentence.encode('ascii', errors='ignore').strip().decode('ascii')

        tokenized_doc = word_tokenize(sentence)
        tokenized_doc = [w for w in tokenized_doc if not w in stopword_add_list][:512]

        return ' '.join(tokenized_doc)

    def predict(self, s_doc):
        
        clean_content = self.clean_sentence(s_doc)

        t5_summary = self.summarizer(clean_content)
        tr_summary = summarize(clean_content, ratio = 0.1)

        return {
	    "general": {
		"word_count" : len(str(tr_summary).split()),
		"summary" : tr_summary 
	    },
	    "finance": {
		"word_count" : len(str(t5_summary[0]['summary_text']).split()),
		"summary" : t5_summary[0]['summary_text']
	    }
        }
        #return t5_summary[0]['summary_text']
