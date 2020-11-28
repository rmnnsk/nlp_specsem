import pandas as pd
import re
from collections import defaultdict
import sys
import numpy as np
from math import log
from scipy.sparse import csr_matrix, dok_matrix
import string
import matplotlib.pyplot as plt

def preprocessing(text):
    text = text.lower()
    remove_tags = re.compile(r'<.*?>')
    text = re.sub(remove_tags, '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ''.join(sym if (sym.isalnum() or sym in (" ", "'")) else f" {sym} " for sym in text)
    return text

def tokenize_dataset(dataset):
    """
        arg: list of texts
        return: list of tokenized texts
    """
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', \
    'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
     'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    tokenizer = re.compile(r"-?\d*[.,]?\d+|[?'\w]+|\S", re.MULTILINE | re.IGNORECASE)
    tokenized_dataset = list(map(lambda doc: tokenizer.findall(doc), dataset))
    stem_dataset = [[token for token in text if token not in stop_words] for text in tokenized_dataset]
    return stem_dataset




train_texts_path = "./filimdb_evaluation/FILIMDB/train.texts"
train_labels_path = "./filimdb_evaluation/FILIMDB/train.labels"

with open(train_texts_path, 'r', encoding='utf-8',) as inp:
    train_texts = list(map(str.strip, inp.readlines()))
with open(train_labels_path, 'r', encoding='utf-8',) as inp:
    train_labels = list(map(str.strip, inp.readlines()))
    

proc_train = list(map(preprocessing, train_texts))
token_train = tokenize_dataset(proc_train)

y_train = np.array([int(lab == 'pos') for lab in train_labels])


def gen_w2ind(tokenized_texts, bigrams=False, trigrams=False):
    w2ind = defaultdict(int)
    free_ind = 0
    for text in tokenized_texts:
        tokens = text[:]
        if bigrams:
            tokens += list(zip(text[:-1], text[1:]))
        if trigrams:
            tokens += list(zip(text[:-2], text[1:-1], text[2:]))
        for token in tokens:
            if token not in w2ind:
                w2ind[token] = free_ind
                free_ind += 1
    return w2ind

# print(token_train[0])

w2_ind_23 = gen_w2ind(token_train, bigrams=False, trigrams=False)
print(len(w2_ind_23))
