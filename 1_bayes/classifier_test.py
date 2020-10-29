import re
from collections import defaultdict
import sys
from math import log, e
import itertools
import random

def preprocessing(text):
    text = text.lower()
    remove_tags = re.compile(r'<.*?>')
    text = re.sub(remove_tags, '', text)
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
    tokenizer = re.compile(r"-?\d*[.,]?\d+|([?'\w]+)|(\S)", re.MULTILINE | re.IGNORECASE)
    tokenized_dataset = list(map(lambda doc: tokenizer.findall(doc), dataset))
    stem_dataset = [[token for token in text if token not in stop_words] for text in tokenized_dataset]
    return stem_dataset



free_ind = 0

def train(texts, labels, max_df=0.1, min_df=1, freq_treshhold=1):
    word_to_ind = {}
    free_ind = 0

    cnt_words = [defaultdict(int), defaultdict(int)]
    cnt_words_in_text = [defaultdict(int), defaultdict(int)]
    df = defaultdict(int)
    df_cnt = defaultdict(int)
    w_freq = defaultdict(int)
    N_words = -1
    prob_w_cls = []
    prob_cls = []
    bigrams = True
    trigrams = True

    proc_dataset = list(map(preprocessing, texts))
    token_dataset = tokenize_dataset(proc_dataset)

    int_labels = [int(lab == 'pos') for lab in labels]

    dataset = list(zip(token_dataset, int_labels))

    for text, lab in dataset:
        been = [set(), set()]
        for word in text:
            if word not in word_to_ind:
                word_to_ind[word] = free_ind
                free_ind += 1
            ind = word_to_ind[word]
            if word not in been[lab]:
                cnt_words_in_text[lab][ind] += 1
                been[lab].add(word)
            cnt_words[lab][ind] += 1

    if bigrams:
        for text, lab in dataset:
            bigr = [b for b in zip(text[:-1], text[1:])]
            been = [set(), set()]
            for word in bigr:
                if word not in word_to_ind:
                    word_to_ind[word] = free_ind
                    free_ind += 1
                ind = word_to_ind[word]
                if word not in been[lab]:
                    cnt_words_in_text[lab][ind] += 1
                    been[lab].add(word)
                cnt_words[lab][ind] += 1

    if trigrams:
        for text, lab in dataset:
            trigr = [b for b in zip(text[:-2], text[1:-1], text[2:])]
            been = [set(), set()]
            for word in trigr:
                if word not in word_to_ind:
                    word_to_ind[word] = free_ind
                    free_ind += 1
                ind = word_to_ind[word]
                if word not in been[lab]:
                    cnt_words_in_text[lab][ind] += 1
                    been[lab].add(word)
                cnt_words[lab][ind] += 1


    cnt_w_cls = [sum(cnt_words[cls].values()) for cls in (0, 1)]
    def cnt_prob_word_cls(ind, cls, alpha=1):
        cnt = cnt_words[cls][ind] if ind != -1 else 0
        return (alpha + cnt) / (cnt_w_cls[cls] + alpha*len(word_to_ind))

    cnt_cls = [sum(1 for text, lab in dataset if lab == cls) for cls in (0, 1)]
    def cnt_prob_cls(cls):
    # 0 - neg, 1 - pos
        return cnt_cls[cls] / sum(cnt_cls)


    # Train with unlabeled data
    #--------------------------------------

    #--------------------------------------

    for w, ind in word_to_ind.items():
        df[ind] = (cnt_words_in_text[0][ind] + cnt_words_in_text[1][ind]) / len(dataset)
        df_cnt[ind] = (cnt_words_in_text[0][ind] + cnt_words_in_text[1][ind])
    
    all_tokens = sum(sum(cnt_words[cls].values()) for cls in (0, 1))
    for w, ind in word_to_ind.items():
        w_freq[ind] = (cnt_words[0][ind] + cnt_words[1][ind]) / all_tokens

    # Deleting words with freq > freq_treshhold.
    for w, ind in list(word_to_ind.items()):
        if df[ind] > max_df or w_freq[ind] > freq_treshhold or df_cnt[ind] < min_df:
            cnt_words[0].pop(ind, -1), cnt_words[1].pop(ind, -1)
            cnt_words_in_text[0].pop(ind, -1), cnt_words_in_text[1].pop(ind, -1)
            word_to_ind.pop(w, -1)
    
    new_word_to_ind = {}
    new_cnt_words = [defaultdict(int), defaultdict(int)]
    new_cnt_words_in_text = [defaultdict(int), defaultdict(int)]
    new_df_cnt = {}
    new_w_freq = {}
    for ind, (w, old_ind) in enumerate(word_to_ind.items()):
        new_word_to_ind[w] = ind
        new_df_cnt[ind] = df_cnt[old_ind]
        new_w_freq[ind] = df[old_ind]
        for cls in (0, 1):
            new_cnt_words_in_text[cls][ind] = cnt_words_in_text[cls][old_ind]
            new_cnt_words[cls][ind] = cnt_words[cls][old_ind]
    
    word_to_ind = new_word_to_ind
    cnt_words_in_text = new_cnt_words_in_text
    cnt_words = new_cnt_words
    df_cnt = new_df_cnt
    w_freq = new_w_freq

    N_words = len(word_to_ind)
    prob_w_cls = [[cnt_prob_word_cls(i, cls) for i in range(N_words)] for cls in (0, 1)]
    prob_cls = [cnt_prob_cls(cls) for cls in (0, 1)]

    return word_to_ind, df_cnt, prob_w_cls, prob_cls, bigrams, trigrams

def vectorize(tokenized_texts, word_to_ind, df_cnt, mode, bigrams=True, trigrams=True):
    res = [defaultdict(int) for _ in range(len(tokenized_texts))]
    for ind, text in enumerate(tokenized_texts):
        for token in text:
            if token in word_to_ind:
                res[ind][word_to_ind[token]] += 1
    if bigrams:
        for ind, text in enumerate(tokenized_texts):
            bigr = [b for b in zip(text[:-1], text[1:])]
            for token in bigr:
                if token in word_to_ind:
                    res[ind][word_to_ind[token]] += 1
    if trigrams:
        for ind, text in enumerate(tokenized_texts):
            trigr = [b for b in zip(text[:-2], text[1:-1], text[2:])]
            for token in trigr:
                if token in word_to_ind:
                    res[ind][word_to_ind[token]] += 1
    if mode == 'bern':
        for text in res:
            for ind, cnt in text.items():
                if cnt > 0:
                    text[ind] = 1
    if mode == 'tfidf':
        for text in res:
            for ind, cnt in text.items():
                text[ind] /= len(text)
                idf = log((1 + len(tokenized_texts))/(1 + df_cnt[ind])  + 1)
                text[ind] *= idf
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
    return res

def classify(dataset, params, unlabeled_data=None):
    #preprocessing
    word_to_ind, df, prob_w_cls, prob_cls, bigr, trigr = params
    dataset = list(map(preprocessing, dataset))
    dataset = tokenize_dataset(dataset)
    dataset = vectorize(dataset, word_to_ind, df, mode='tfidf', bigrams=bigr, trigrams=trigr)
    probs = []
    size = len(dataset)
    for i, text in enumerate(dataset):
        if i % 1000 == 0:
            print(f'complited: {i}/{size}')
        probs.append(classify_mult(text, prob_w_cls, prob_cls))
    res = ['pos' if pr[1] > pr[0] else 'neg' for pr in probs]
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

    #Train with unlabeled
