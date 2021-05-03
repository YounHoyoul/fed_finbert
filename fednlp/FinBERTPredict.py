import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import pandas as pd
import numpy as np
import re
import spacy
import math
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, confusion_matrix
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertConfig

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class BertClassification(nn.Module):
    def __init__(self, weight_path, num_labels=3, vocab="finance-uncased"):
        super(BertClassification, self).__init__()
        self.num_labels = num_labels
        self.vocab = vocab 
        if self.vocab =="finance-uncased":
            self.bert = BertModel.from_pretrained(weight_path)
            self.config = BertConfig(vocab_size_or_config_json_file=30873, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        nn.init.xavier_normal(self.classifier.weight)
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, graphEmbeddings=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
       
        logits = self.classifier(pooled_output)
            
        return logits

class DenseOptim():
    def __init__(self, model):
        super(DenseOptim, self).__init__()
        self.lrlast = .001
        self.lrmain = .00001

        self.optim = optim.Adam([ 
            {
                "params":model.bert.parameters(),
                "lr": self.lrmain
            },
            {
                "params":model.classifier.parameters(), 
                "lr": self.lrlast
            },
        ])
    
    def get_optim(self):
        return self.optim

class FedPredictDataset(Dataset):
    def __init__(self, texts, vocab_path, max_seq_length=512, vocab = 'finance-uncased'):
        self.texts = texts;
        self.dict_labels = {'lower': 0, 'maintain':1, 'raise':2}

        self.max_seq_length = max_seq_length
        self.vocab = vocab
        if self.vocab == 'finance-uncased':
            self.tokenizer = BertTokenizer(vocab_file = vocab_path, do_lower_case = True, do_basic_tokenize = True)

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        tokenized_review = self.tokenizer.tokenize(self.texts[index])

        if len(tokenized_review) > self.max_seq_length:
            tokenized_review = tokenized_review[:self.max_seq_length]

        ids_review  = self.tokenizer.convert_tokens_to_ids(tokenized_review)

        mask_input = [1]*len(ids_review)
        
        padding = [0] * (self.max_seq_length - len(ids_review))
        ids_review += padding
        mask_input += padding
        
        input_type = [0]*self.max_seq_length  
        
        assert len(ids_review) == self.max_seq_length
        assert len(mask_input) == self.max_seq_length
        assert len(input_type) == self.max_seq_length 
        
        ids_review = torch.tensor(ids_review)
        mask_input =  torch.tensor(mask_input)
        input_type = torch.tensor(input_type)

        input_feature = {"token_type_ids": input_type, "attention_mask":mask_input, "input_ids":ids_review}    

        return input_feature

class PredictModel:
    def __init__(self, model_type, output_path):
        self.output_path = output_path
        self.dict_indexes = {0: 'lower', 1 : 'maintain', 2 : 'raise'}
        
        self.model_path = os.path.join(output_path, "{}.pth".format(model_type))

    def predict(self, model, text, vocab_path, vocab, device, batch_size = 8):
        outputs = self.predict_texts(model, [text], vocab_path, vocab, device, batch_size)

        indexes = torch.argmax(outputs, dim=1)
        indexes = indexes.cpu().numpy()
        labels = []

        for index in indexes:
            labels.append(self.dict_indexes[index])

        return labels, outputs

    def predict_texts(self, model, texts, vocab_path, vocab, device, batch_size = 8):
        model.to(device)

        map_location=torch.device('cpu')
        
        model.load_state_dict(torch.load(self.model_path, map_location=map_location))
        model.eval()

        predict_dataset = FedPredictDataset(texts, vocab_path=vocab_path, vocab=vocab)
        dataloader = torch.utils.data.DataLoader(predict_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        result = torch.tensor([]).long().to(device)

        for inputs in tqdm(dataloader):
            input_ids = inputs["input_ids"].to(device)
            token_type_ids = inputs["token_type_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            with torch.set_grad_enabled(False):
                  outputs = model(input_ids, token_type_ids, attention_mask)
                  outputs = F.softmax(outputs, dim=1)
                  result = torch.cat([result, outputs], dim=0)

        return result 
        