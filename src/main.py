from flask import Flask
from flask import request, jsonify
from flask_cors import CORS, cross_origin
import torch
import json
import numpy
from modules.SentimentPredict import SentimentPredict
from modules.WordCloudPredict import WordCloudPredict
from modules.SummarizationPredict import SummarizationPredict
from modules.FinBERTPredict import PredictModel, BertClassification
from modules.ExplainPredict import ExplainPredict

app = Flask(__name__)
cors = CORS(app)

VOCAB = 'finance-uncased'
VOCAB_PATH = '/app/output/FinVocab-Uncased.txt'
WEIGHT_PATH = '/app/output/FinBERT-FinVocab-Uncased.tar.gz'
OUTPUT_PATH = '/app/output'
BATCH_SIZE = 8
NUM_LABELS = 3

sentiment = SentimentPredict()
wordcloud = WordCloudPredict()
summarization = SummarizationPredict()
prediction_model = BertClassification(weight_path=WEIGHT_PATH, num_labels=NUM_LABELS, vocab=VOCAB)
explain = ExplainPredict()

def convert(o):
    if isinstance(o, numpy.int64): return int(o)  
    raise TypeError

@app.route('/sentiment', methods = ['POST'])
def predict_sentiment():
    result = sentiment.predict(request.form.get('content'))

    return json.dumps(result, sort_keys=True, default=convert)

@app.route('/wordcloud', methods = ['POST'])
def predict_wordcloud():
    result = wordcloud.predict(request.form.get('content'))

    return {
        "image:": "base64:" + result.decode('utf-8')
    }

@app.route('/summary', methods = ['POST'])
def predict_summary():
    result = summarization.predict(request.form.get('content'))

    return result

@app.route('/prediction', methods = ['POST'])
def predict_prediction():
    s_doc = request.form.get('content')

    index, _ = PredictModel(model_type='checkpoint_up', output_path=OUTPUT_PATH).predict(
        prediction_model, 
        s_doc, 
        vocab_path = VOCAB_PATH, 
        vocab = VOCAB,
        device = "cpu",
        batch_size = BATCH_SIZE
    )

    return {
        "prediction:": index[0]
    }

@app.route('/explain', methods = ['POST'])
def predict_explain():
    result = explain.predict(request.form.get('content'))

    return {
        "explain:": result
    }

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
