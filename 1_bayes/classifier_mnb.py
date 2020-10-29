import re
from collections import defaultdict
import sys
from math import log


def preprocessing(text):
    text = text.lower()
    remove_tags = re.compile('<.*?>')
    text = re.sub(remove_tags, '', text)
    text = ''.join(sym if (sym.isalnum() or sym in (" ", )) else f" {sym} " for sym in text)
    return text


def tokenize_dataset(dataset):
    """
        arg: list of texts
        return: list of tokenized texts
    """
    tokenizer = re.compile("-?\d*[.,]?\d+|[\w]+|\S", re.MULTILINE | re.IGNORECASE)
    return list(map(lambda doc: tokenizer.findall(doc), dataset))



free_ind = 0

def train(dataset, labels):
    word_to_ind = {}
    free_ind = 0

    pos_words = defaultdict(int)
    word_in_pos_text = defaultdict(int)
    neg_words = defaultdict(int)
    word_in_neg_text = defaultdict(int)
    cnt_pos = 0
    cnt_neg = 0
    N_words = -1
    prob_w_cls = []
    prob_cls = []
    bigrams = True
    trigrams = True
    #--------------------------------------
    train_pos = [dataset[i] for i, lab in enumerate(labels) if lab == 'pos']
    train_pos = list(map(preprocessing, train_pos))
    train_pos = tokenize_dataset(train_pos)

    train_neg = [dataset[i] for i, lab in enumerate(labels) if lab == 'neg']
    train_neg = list(map(preprocessing, train_neg))
    train_neg = tokenize_dataset(train_neg)

    for text in train_pos:
        been = set()
        for word in text:
            if word not in word_to_ind:
                word_to_ind[word] = free_ind
                free_ind += 1
            if word not in been:
                ind = word_to_ind[word]
                word_in_pos_text[ind] += 1
                been.add(word)
            pos_words[word_to_ind[word]] += 1
            cnt_pos += 1

    if bigrams:
        for text in train_pos:
            been = set()
            bigr = [b for b in zip(text[:-1], text[1:])]
            for word in bigr:
                if word not in word_to_ind:
                    word_to_ind[word] = free_ind
                    free_ind += 1
                if word not in been:
                    ind = word_to_ind[word]
                    word_in_pos_text[ind] += 1
                    been.add(word)
                pos_words[word_to_ind[word]] += 1
                cnt_pos += 1

    if trigrams:
        for text in train_pos:
            been = set()
            trigr = [b for b in zip(text[:-2], text[1:-1], text[2:])]
            for word in trigr:
                if word not in word_to_ind:
                    word_to_ind[word] = free_ind
                    free_ind += 1
                if word not in been:
                    ind = word_to_ind[word]
                    word_in_pos_text[ind] += 1
                    been.add(word)
                pos_words[word_to_ind[word]] += 1
                cnt_pos += 1

    for text in train_neg:
        been = set()
        for word in text:
            if word not in word_to_ind:
                word_to_ind[word] = free_ind
                free_ind += 1
            if word not in been:
                ind = word_to_ind[word]
                word_in_neg_text[ind] += 1
                been.add(word)    
            neg_words[word_to_ind[word]] += 1
            cnt_neg += 1

    if bigrams:
        for text in train_neg:
            been = set()
            bigr = [b for b in zip(text[:-1], text[1:])]
            for word in bigr:
                if word not in word_to_ind:
                    word_to_ind[word] = free_ind
                    free_ind += 1
                if word not in been:
                    ind = word_to_ind[word]
                    word_in_neg_text[ind] += 1
                    been.add(word)
                neg_words[word_to_ind[word]] += 1
                cnt_neg += 1

    if trigrams:
        for text in train_neg:
            been = set()
            trigr = [b for b in zip(text[:-2], text[1:-1], text[2:])]
            for word in trigr:
                if word not in word_to_ind:
                    word_to_ind[word] = free_ind
                    free_ind += 1
                if word not in been:
                    ind = word_to_ind[word]
                    word_in_neg_text[ind] += 1
                    been.add(word)
                neg_words[word_to_ind[word]] += 1
                cnt_neg += 1


    N_words = len(word_to_ind)

    def cnt_prob_word_cls(ind, cls):
        if cls == 'neg':
            cnt = neg_words[ind] if ind != -1 else 0
            return (1 + cnt) / (cnt_neg + len(word_to_ind))
        else:
            cnt = pos_words[ind] if ind != -1 else 0
            return (1 + cnt) / (cnt_pos + len(word_to_ind))

    def cnt_prob_cls(cls):
    # 0 - neg, 1 - pos
        if cls == 0:
            return len(train_neg) / (len(train_neg) + len(train_pos))
        else:
            return len(train_pos) / (len(train_neg) + len(train_pos))

    prob_w_cls = [[cnt_prob_word_cls(i, cls) for i in range(N_words)] + [cnt_prob_word_cls(-1, cls)] for cls in ('neg', 'pos')]
    prob_cls = [cnt_prob_cls(cls) for cls in ('neg', 'pos')]

    return word_to_ind, prob_w_cls, prob_cls, bigrams

def vectorize(tokenized_texts, word_to_ind, mode, bigrams=True, trigrams=True):
    res = [defaultdict(int) for _ in range(len(tokenized_texts))]
    for ind, text in enumerate(tokenized_texts):
        for token in text:
            if token in word_to_ind:
                res[ind][word_to_ind[token]] += 1
            else:
                res[ind][-1] += 1
    if bigrams:
        for ind, text in enumerate(tokenized_texts):
            bigrams = [b for b in zip(text[:-1], text[1:])]
            for token in bigrams:
                if token in word_to_ind:
                    res[ind][word_to_ind[token]] += 1
                else:
                    res[ind][-1] += 1
    if trigrams:
        for ind, text in enumerate(tokenized_texts):
            trigr = [b for b in zip(text[:-2], text[1:-1], text[2:])]
            for token in trigr:
                if token in word_to_ind:
                    res[ind][word_to_ind[token]] += 1
    if mode == 'bern':
        for text in res:
            for k, v in text.items():
                if v > 0:
                    text[k] = 1
    return res

def classify_mult(vectorized_text, prob_w_cls, prob_cls):
    # 0 - neg, 1 - pos
    classes = (0, 1)
    res = []
    for cls in classes:
        cur_res = 0
        for ind, word in enumerate(vectorized_text.keys()):
            cur_res += log(prob_w_cls[cls][word]) * vectorized_text[word]
        cur_res += log(prob_cls[cls])
        res.append(cur_res)
    if res[0] > res[1]:
        label = 0
    else:
        label = 1
    return label

def classify(dataset, params):
    #preprocessing
    word_to_ind, prob_w_cls, prob_cls, bigr = params
    dataset = list(map(preprocessing, dataset))
    dataset = tokenize_dataset(dataset)
    dataset = vectorize(dataset, word_to_ind, mode='mult', bigrams=bigr)
    probs = []
    size = len(dataset)
    for i, text in enumerate(dataset):
        if i % 1000 == 0:
            print(f'complited: {i}/{size}')
        probs.append(classify_mult(text, prob_w_cls, prob_cls))
    res = ['pos' if pr else 'neg' for pr in probs]
    return res

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