import score
import re
import numpy as np
import pickle
from numpy import linalg as LA
import time
import matplotlib.pyplot as plt


def init_params(layer_sizes, activation, use_previous=False):
    if use_previous:
        with open('W[200, 100, 20, 10]30.pickle', 'rb') as f:
            W = pickle.load(f)
    else:
        W = {}
        for i in range(1, len(layer_sizes)):
            W['W'+str(i)] = np.random.randn(layer_sizes[i-1] + 1, layer_sizes[i]) * np.sqrt(2.0/(layer_sizes[i-1] + 1))
    return W


def fully_connected(a_prev, W, activation):
    Z = np.matmul(np.append(np.ones((a_prev.shape[0], 1)), a_prev, axis=1), W)
    if activation == 'tanh':
        A = np.tanh(Z)
    else:
        A = np.tanh(Z)
    return A, Z


def ffnn(X, params, activation):
    Z_cache = []
    A = X
    A_cache = []
    A_cache.append(X)
    for i in range(len(params)):
        A, zi = fully_connected(A, params['W'+str(i+1)], activation)
        Z_cache.append(zi)
        A_cache.append(A)

    return zi, Z_cache, A_cache


def softmax_crossentropy(ZL, Y):
    N = ZL.shape[0]
    Exp = np.exp(ZL - np.max(ZL, axis=1).reshape((ZL.shape[0], 1)))
    AL = Exp/(np.sum(Exp, axis=1)).reshape((Exp.shape[0], 1))
    CE = -np.sum(np.log(AL[Y])) / N
    return CE, AL


def fully_connected_backward(delta_cur, W, A_prev, Z, activation):
    W = np.delete(W, 0, 0)
    delta_prev = np.matmul(delta_cur, W.T) * (1 - np.tanh(Z) ** 2)
    grad_W_l = np.matmul((np.append(np.ones((A_prev.shape[0], 1)), A_prev, axis=1)).T, delta_cur)
    return delta_prev, grad_W_l


def ffnn_backward(dZL, Z_cache, A_cache, W, activation):
    dZ = dZL
    grad = {}
    for i in reversed(range(1, len(W))):
        dZ, grad['W' + str(i + 1)] = fully_connected_backward(dZ, W['W' + str(i+1)], A_cache[i], Z_cache[i-1], 'tanh')
    grad['W1'] = np.matmul((np.append(np.ones((A_cache[0].shape[0], 1)), A_cache[0], axis=1)).T, dZ)
    return grad


def softmax_crossentropy_backward(ZL, Y):
    N = ZL.shape[0]
    Exp = np.exp(ZL - np.max(ZL, axis=1).reshape((ZL.shape[0], 1)))
    softmax = Exp / (np.sum(Exp, axis=1)).reshape((Exp.shape[0], 1))
    dZL = 1/N * (softmax - Y)
    return dZL


def sgd_step(W, grads, learning_rate, alpha):
    for i in W.keys():
        W[i] = W[i] - learning_rate * grads[i] - 2*learning_rate*alpha*W[i]
    return W


def ada_step(W, grads, learning_rate, alpha, squares, eps=1e-8):
    for i in W.keys():
        squares[i] = squares[i] + grads[i] ** 2
        W[i] = W[i] - learning_rate * grads[i] / (squares[i] + eps) ** 0.5 - 2*learning_rate*alpha*W[i]
        W[i][0] = W[i][0] + 2*learning_rate*alpha*W[i][0]
    return W, squares


def make_embeddings(compute_again=True):
    if compute_again:
        with open("glove.6B.300d.txt", "r") as f:
            data = f.readlines()
        i = 0
        words = {}
        Embeds = np.empty((len(data), len(data[0].split(' ')) - 1), dtype=np.double)
        print('Computing Embeddings')
        for line in data:
            splitted = line.split(" ")
            words[splitted[0]] = i
            Embeds[i] = (splitted[1:])
            i = i + 1

        # Embeds = Embeds / (LA.norm(Embeds, axis=1).reshape(Embeds.shape[0], 1))
        with open('E.pickle', 'wb') as Efile:
            pickle.dump(Embeds, Efile)
        with open('words.pickle', 'wb') as  Vfile:
            pickle.dump(words, Vfile)
    else:
        with open('E.pickle', 'rb') as Efile:
            Embeds = pickle.load(Efile)
        with open('words.pickle', 'rb') as Vfile:
            words = pickle.load(Vfile)

    return Embeds, words


def preprocess(train_texts, Embeds, words):

    p = re.compile(r'\W+')    
    M = Embeds.shape[1]
    vocab = set(words.keys())

    print('Tokenization...')
    XMatrix = np.empty((len(train_texts), M))
    # nfound = 0
    # nvocab = set()
    for i in range(len(train_texts)):
        doc = p.split((train_texts[i].lower()).strip())
        Vec = np.zeros(M, )
        count = 0
        for word in doc:
            if word in vocab:
                j = words[word]
                Vec = Vec + Embeds[j]
                count = count + 1
            # else:
            #     if word != '':
            #         nfound = nfound + 1
            #         nvocab.add(word)
            #         if nfound <= 25:
            #             print(word+',')

        Vec = Vec / count
        XMatrix[i] = Vec

    return XMatrix


def labels_processing(train_labels):
    Y = np.array(train_labels)
    Y = Y.reshape((Y.shape[0], 1))
    Y = Y == 'neg'
    Y = np.append(Y, ~Y, axis=1)
    return Y


def train(train_texts, train_labels, learning_rate = 1e-2, epochs = 1001, batch_size = 1500, alpha = 2e-3,
          layers_size=None):
    """
    Trains classifier on the given train set represented as parallel lists of texts and corresponding labels.
    :param train_texts: a list of texts (str objects), one str per example
    :param train_labels: a list of labels, one label per example
    :return: learnt parameters, or any object you like (it will be passed to the classify function)
    """

    if layers_size is None:
        layers_size = [100, 10]
    squares = {}
    ceVec = []
    compute_again = True
    if compute_again:
        Embeds, words = make_embeddings(False)
        X = preprocess(train_texts, Embeds, words)
        Y = labels_processing(train_labels)

        with open('X.pickle', 'wb') as Xfile:
            pickle.dump(X, Xfile)
        with open('Y.pickle', 'wb') as Yfile:
            pickle.dump(Y, Yfile)
    else:
        with open('X.pickle', 'rb') as Xfile:
            X = pickle.load(Xfile)
        with open('Y.pickle', 'rb') as Yfile:
            Y = pickle.load(Yfile)

    print('preprocessing ---> Done!')

    ls = layers_size.copy()
    ls.append(2)
    ls.insert(0, X.shape[1])
    iterations_per_epoch = X.shape[0]/batch_size
    iterations = round(epochs * iterations_per_epoch)

    W = init_params(ls, 'tanh', use_previous=True)

    #For adagrad
    for k in W.keys():
        squares[k] = np.zeros(W[k].shape)

    x_size = X.shape[0]
    for i in range(iterations):
        left = (i*batch_size) % x_size
        right = left + batch_size

        ZL, Z_cache, A_cache = ffnn(X[left:right], W, 'tanh')
        CE, AL = softmax_crossentropy(ZL, Y[left:right])
        cur_epoch = round(i / iterations_per_epoch)
        if (cur_epoch % 200 == 0) & (i % iterations_per_epoch == 0):
            print('Epoch: ' + str(cur_epoch))
            print(CE)
            ceVec.append(CE)

        # print(np.mean(AL, axis=0))
        # # --------------Gradient checking-----------------
        # eps = 1e-3
        # w_name = 'W1'
        # CE_right = np.empty(W[w_name].shape)
        # CE_left = np.empty(W[w_name].shape)
        # for i in range(W[w_name].shape[0]):
        #     for j in range(W[w_name].shape[1]):
        #         W[w_name][i, j] = W[w_name][i, j] + eps
        #         ZL1, Z_cache1, A_cache1 = ffnn(X, W, 'tanh')
        #         CE_right[i, j], AL1 = softmax_crossentropy(ZL1, Y)
        #
        #         W[w_name][i, j] = W[w_name][i, j] - 2*eps
        #         ZL1, Z_cache1, A_cache1 = ffnn(X, W, 'tanh')
        #         CE_left[i, j], AL1 = softmax_crossentropy(ZL1, Y)
        #
        #         W[w_name][i, j] = W[w_name][i, j] + eps
        #
        # Num_grad_W3 = (CE_right - CE_left)/(2 * eps)
        #
        dZL = softmax_crossentropy_backward(ZL, Y[left:right])
        grad = ffnn_backward(dZL, Z_cache, A_cache, W, 'tanh')
        # print(grad[w_name] - Num_grad_W3)

        W = sgd_step(W, grad, learning_rate, alpha)
        # W, squares = ada_step(W, grad, learning_rate, alpha, squares)
    W['Embeds'] = Embeds
    W['words'] = words
    return W, ceVec


def classify(texts, params):
    """
    Classify texts given previously learnt parameters.
    :param texts: texts to classify
    :param params: parameters received from train function
    :return: list of labels corresponding to the given list of texts
    """

    Embeds = params.pop('Embeds')
    words = params.pop('words')

    X = preprocess(texts, Embeds, words)
    ZL, Z_cache, A_cache = ffnn(X, params, 'tanh')
    Exp = np.exp(ZL - np.max(ZL, axis=1).reshape((ZL.shape[0], 1)))
    softmax = Exp / (np.sum(Exp, axis=1)).reshape((Exp.shape[0], 1))
    predictions = softmax
    labels = list(map(lambda x: 'pos' if x == 1 else 'neg', list(np.round(predictions[:, 1]))))
    params['Embeds'] = Embeds
    params['words'] = words
    return labels

# text = score.load_dataset_fast()
# train_ids, train_texts, train_labels = text['train']
# test_ids, test_texts, test_labels = text['dev']
# params = train(train_texts[:], train_labels[:])
# with open("W.pickle", 'wb') as f:
#     pickle.dump(params, f)
#
# labels = classify(train_texts[:], params)
#
# real_labels = np.array(list(map(lambda x: int(x == 'pos'), train_labels[:])))
# predicted_labels = np.array(list(map(lambda x: int(x == 'pos'), labels)))
# accuracy = 1 - np.sum(np.abs(real_labels - predicted_labels)) / len(real_labels)
# print(accuracy)