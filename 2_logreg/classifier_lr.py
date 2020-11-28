import re
from collections import defaultdict
import sys
import numpy as np
from math import log
from scipy.sparse import csr_matrix, dok_matrix
import string


def preprocessing(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = ''.join(sym if (sym.isalnum() or sym in (" ", "'")) else f" {sym} " for sym in text)
    return text


def tokenize_dataset(dataset, stem=0):
    """
        arg: list of texts
        return: list of tokenized texts
    """
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
                  "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's",
                  'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
                  'themselves', 'what', 'which', 'who', 'whom', 'this',
                  'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
                  'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
                  'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
                  'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
                  'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
                  'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
                  'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
                  't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',
                  've', 'y', 'ain', 'aren', "aren't", 'could', 'couldn', "couldn't", 'didn', "didn't", 'doesn',
                  "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn',
                  "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn',
                  "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"] + [c for c in string.punctuation]
    tokenizer = re.compile(r"-?\d*[.,]?\d+|[?'\w]+|\S", re.MULTILINE | re.IGNORECASE)
    tokenized_dataset = list(map(lambda doc: tokenizer.findall(doc), dataset))
    if stem == 0:
        return [[token for token in text if token not in stop_words] for text in tokenized_dataset]
    stem_dataset = [[token[:stem] for token in text if token not in stop_words] for text in tokenized_dataset]
    return stem_dataset


def gen_w2ind(tokenized_texts, bigrams=False, trigrams=False):
    w2ind = defaultdict(int)
    df_cnt = defaultdict(int)
    ind2w = {}
    free_ind = 0
    for text in tokenized_texts:
        been = set()
        tokens = text[:]
        if bigrams:
            tokens += list(zip(text[:-1], text[1:]))
        if trigrams:
            tokens += list(zip(text[:-2], text[1:-1], text[2:]))
        for token in tokens:
            if token not in w2ind:
                w2ind[token] = free_ind
                ind2w[free_ind] = token
                free_ind += 1
            if w2ind[token] not in been:
                df_cnt[w2ind[token]] += 1
                been.add(w2ind[token])
    return w2ind, ind2w, df_cnt


def vectorize(tokenized_texts, w2ind, bigrams=False, trigrams=False, intercept=True):
    X = dok_matrix((len(tokenized_texts), len(w2ind) + intercept), dtype=np.float32)
    print("test:", X.shape)
    for ind, text in enumerate(tokenized_texts):
        tokens = text[:]
        if bigrams:
            tokens += list(zip(text[:-1], text[1:]))
        if trigrams:
            tokens += list(zip(text[:-2], text[1:-1], text[2:]))
        for token in tokens:
            if token in w2ind:
                token_ind = w2ind[token]
                if intercept:
                    token_ind += 1
                X[ind, token_ind] += 1
    for ind, text in enumerate(tokenized_texts):
        if intercept:
            X[ind, 0] = 1
    X = X.tocsr()
    return X


def batch_generator(X, y, shuffle=True, batch_size=2):
    """
    Генератор новых батчей для обучения
    """

    X_all = X
    y_all = y
    indices = np.arange(X_all.shape[0])
    if shuffle:
        np.random.shuffle(indices)
    for i in range(0, X_all.shape[0], batch_size):
        if len(indices[i: i + batch_size]) != batch_size:
            break
        X_batch = X_all[indices[i: i + batch_size]]
        y_batch = y_all[indices[i: i + batch_size]]
        yield (X_batch, y_batch)


def sigmoid(x):
    np.clip(x, -100, 100)
    sig = 1 / (1 + np.exp(-x))
    sig = np.minimum(sig, 1.0 - np.finfo(np.float32).eps)
    sig = np.maximum(sig, 0.0 + np.finfo(np.float32).eps)
    return sig


class LogisticRegression:
    def __init__(self, num_ep, lr, batch_size, l2_reg, optimizer='SGD', beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.num_ep = num_ep
        self.lr = lr
        self.batch_size = batch_size
        self.l2_reg = l2_reg
        self.optimizer = optimizer
        self.batch_generator = batch_generator
        self.weights = None
        self.m = None
        self.v = None
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.train_info = ([], [], [])
        self.dev_info = ([], [], [])

    def update_w_def(self, new_grad):
        self.weights = self.weights - self.lr * new_grad

    def update_w_adam(self, new_grad, it):
        self.m = self.beta1 * self.m + (1. - self.beta1) * new_grad
        self.mt = self.m / (1 - self.beta1 ** it)
        self.v = self.beta2 * self.v + (1. - self.beta2) * new_grad ** 2
        self.vt = self.v / (1 - self.beta1 ** it)
        self.weights = self.weights - self.lr * self.mt / (np.sqrt(self.vt) + self.epsilon)

    def update_weights(self, data):
        if self.optimizer == 'SGD':
            self.update_w_def(*data[:-1])
        elif self.optimizer == 'Adam':
            self.update_w_adam(*data)

    def loss_grad(self, X, y):
        y_pred = sigmoid(X.dot(self.weights))
        loss = (-y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)).mean()
        loss += self.l2_reg * (self.weights ** 2).mean()

        y_lab = (y_pred > 0.5).astype(np.int32)
        acc = (y_lab == y).sum() / y_lab.shape[0]

        grad = X.T.dot(y_pred - y) / y.shape[0]
        grad += 2 * self.l2_reg * np.insert(self.weights[1:], 0, 0, axis=0)
        return loss, acc, grad

    def fit(self, X, y, dev_dataset=None):

        self.weights = np.zeros(X.shape[1])
        self.m = np.zeros(X.shape[1])
        self.v = np.zeros(X.shape[1])
        total_steps = 1

        for cur_ep in range(0, self.num_ep):
            if cur_ep % 1 == 0:
                loss, acc, _ = self.loss_grad(X, y)
                self.train_info[0].append(cur_ep)
                self.train_info[1].append(loss)
                self.train_info[2].append(acc)
                if dev_dataset:
                    X_dev, y_dev = dev_dataset
                    loss, acc, _ = self.loss_grad(X_dev, y_dev)
                    self.dev_info[0].append(cur_ep)
                    self.dev_info[1].append(loss)
                    self.dev_info[2].append(acc)

            new_epoch_generator = self.batch_generator(X, y, batch_size=self.batch_size, shuffle=True)
            for ind, (X_batch, y_batch) in enumerate(new_epoch_generator):
                loss, acc, grad = self.loss_grad(X_batch, y_batch)
                update_data = [grad, total_steps]
                self.update_weights(update_data)
                total_steps += 1

                # if cur_ep != 0 and cur_ep % 7 == 0:
                #     self.lr *= np.exp(-0.2)
        return self

    def predict(self, X):
        y_pred = sigmoid(X.dot(self.weights))
        y_lab = (y_pred > 0.5).astype(np.int32)
        return y_lab


def train(texts, labels):
    vectorization_params = \
        {
            'bigrams': True,
            'trigrams': False
        }
    y_train = np.array([int(lab == 'pos') for lab in labels])

    proc_train = list(map(preprocessing, texts))
    token_train = tokenize_dataset(proc_train)
    w2ind_train, ind2w_train, df_cnt_train = gen_w2ind(token_train, **vectorization_params)

    X_train = vectorize(token_train, w2ind=w2ind_train, **vectorization_params)

    model = LogisticRegression(num_ep=50, lr=0.2, l2_reg=1e-4, batch_size=1000, optimizer='SGD')

    return model.fit(X_train, y_train), w2ind_train, df_cnt_train, vectorization_params


def classify(texts, params):
    model, w2ind_train, df_cnt_train, vec_params = params

    proc_test = list(map(preprocessing, texts))
    token_test = tokenize_dataset(proc_test)

    X_test = vectorize(token_test, w2ind=w2ind_train, **vec_params)
    y_pred = model.predict(X_test)

    res = ['pos' if pr == 1 else 'neg' for pr in y_pred]

    return res


if __name__ == '__main__':
    def score(my_labels, corr_labels):
        corr = 0
        for pair in zip(my_labels, corr_labels):
            corr += int(pair[0] == pair[1])
        return corr / len(my_labels)


    if __name__ == '__main__':
        train_texts_path = "./filimdb_evaluation/FILIMDB/train.texts"
        train_labels_path = "./filimdb_evaluation/FILIMDB/train.labels"
        with open(train_texts_path, 'r', encoding='utf-8') as inp:
            train_data = list(map(str.strip, inp.readlines()))
        with open(train_labels_path, 'r', encoding='utf-8') as inp:
            train_labels = list(map(str.strip, inp.readlines()))

        train_info = train(train_data, train_labels)

        print("TRAINING_DONE")

        val_texts_path = "./filimdb_evaluation/FILIMDB/dev.texts"
        val_labels_path = "./filimdb_evaluation/FILIMDB/dev.labels"

        with open(val_texts_path, 'r', encoding='utf-8') as inp:
            val_data = list(map(str.strip, inp.readlines()))
        with open(val_labels_path, 'r', encoding='utf-8') as inp:
            val_labels = list(map(str.strip, inp.readlines()))

        my_labels = classify(val_data, train_info)
        print("CLASSIFIED VAL: ", score(my_labels, val_labels))
