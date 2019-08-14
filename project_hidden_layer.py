import numpy as np
import pickle
import scipy
from sklearn.linear_model import Ridge
import time

import matplotlib.pyplot as plt 
from helpers import *


def get_W_diff(y_true, J, residuals, penalty=10000):
    # weights the positive and negative examples equally
    tot_true = np.sum(y_true)
    weighting =  (y_true.shape[0] - tot_true) / tot_true 
    sample_weights = np.ones(y_true.shape)
    # sample_weights[y_true] *= weighting * 1000000
    # positive examples are 97/3 and negative are 1 if 97 neg and 3 pos examples


    clf = Ridge(alpha=penalty, fit_intercept=True)
    clf.fit(J, residuals, sample_weight=sample_weights) 
    diff = clf.coef_
    return diff

data_list, labels = load()
kernel_sz = 3
stride = 2
num_filters = 20
Xs = []
num_weights = kernel_sz**2 * 3

for b_ind in range(5):
    data = data_list[b_ind]
    X = []
    for i in range(0, 32-kernel_sz+1, stride):
        for j in range(0, 32-kernel_sz+1, stride):
            d = data[:, i:i+kernel_sz, j:j+kernel_sz, :].reshape(-1, num_weights)
            X.append(d)
    X = np.array(X).transpose([1, 0, 2])
    Xs.append(X)


W = np.random.randn(kernel_sz**2 * 3, num_filters) / num_filters / num_weights / X.shape[1]

num_convolves = X.shape[1]
alpha = np.random.randn(num_convolves, num_filters)
penalty=10000
loss = []
for j in range(200):
    # penalty *= 1.5
    temp_loss = 0
    # pen = 1.6**(j/10) + 1000*j
    # pen = 1
    
    for i in range(5):
        X = Xs[i][:100]
        y_true = (labels[i] == 1)[:100]
        XW = X @ W
        D = XW > 0

        # print(np.sum(D), D.shape[0] * D.shape[1]*D.shape[2])
        # print(D.shape, X.shape, W.shape, np.sum(D))

        DX = np.expand_dims(D, 2) * np.expand_dims(X, 3) # (num_examples, num_convolves, num_weights, num_filters)
        # multiply by alpha
        alphaDX = np.expand_dims(alpha, 1) * DX # (num_examples, num_convolves, num_weights, num_filters)
        # multiply by weights
        alphaDXW = alphaDX * W # (num_examples, num_convolves, num_weights, num_filters)
        # sum out num_weights
        alphaDXW = np.sum(alphaDXW, axis=2) # (num_examples, num_convolves, num_filters)

        # sum and sigmoid
        preds = sigmoid(np.sum(alphaDXW, axis=(1, 2))) # (num_examples)
        residuals = preds - y_true
        # diffs = np.square(preds - (labels[i] == 1))
        # loss = np.sum(diffs)
        # U = np.random.choice(range(X.shape[0]), size=100, p=diffs/loss)
        # print(U.shape)
        # quit()
        # print("loss", loss)
        # print((labels[i] == 1)[:50])
        # print(preds[:50])
        # print(DX.shape, np.sum(DX * W, axis=2).shape)
        # print(np.sum((np.sum(DX * W, axis=2) > 200)))
        # print((DX)[0, 0] * W)
        # print((DX*W)[0, 0])
        J = np.sum(alphaDX, axis=1) # (num_examples, num_weights, num_filters)
        # print(J.shape, DX.shape)

        J = J.reshape(-1, num_weights* num_filters) # (num_examples, num_weights * num_filters) 
        J = np.expand_dims((preds - y_true) * preds * (1.-preds), 1) * J

        # A = np.concatenate([np.eye(num_weights * num_filters), A])

        # JW_t = J @ W.reshape(-1)
        # b = residuals - JW_t
        # b = (labels[i] == 1)[:100]
        # b = np.concatenate([W.reshape(-1), b])
        # weights = [pen] * num_weights * num_filters + [1] * X.shape[0]
        # W_diff, _, _, _ = np.linalg.lstsq(J, residuals, rcond=None) # (num_weights * num_filters)

        # print(preds[:50])


        W_diff = get_W_diff(y_true, J, -residuals, penalty=penalty)


        # print(np.linalg.norm(W_diff))
        # if np.linalg.norm(W_diff) < .000000000001:

        # print(preds)

        W_diff = W_diff.reshape(kernel_sz**2 * 3, num_filters) # (num_weights, num_filters)

        W_next = W + W_diff
        # if np.linalg.norm(W - temp) < 1e-5:
        #     break
        # print(np.linalg.norm(W - W_next))

        # ne, _, _, _ = np.linalg.lstsq(A, residuals, rcond=None) # use this instead I think
        # a = np.average((W - temp) / ne.reshape(kernel_sz**2 * 3, num_filters))
        # print(np.linalg.norm(W - a * ne.reshape(kernel_sz**2 * 3, num_filters) - temp))
        W = .9 * W + .1 * W_next



        ################################
        XW = X @ W
        D = XW > 0

        # print(np.sum(D), D.shape[0] * D.shape[1]*D.shape[2])
        # print(D.shape, X.shape, W.shape, np.sum(D))

        DX = np.expand_dims(D, 2) * np.expand_dims(X, 3) # (num_examples, num_convolves, num_weights, num_filters)
        # multiply by alpha
        alphaDX = np.expand_dims(alpha, 1) * DX # (num_examples, num_convolves, num_weights, num_filters)
        # multiply by weights
        alphaDXW = alphaDX * W # (num_examples, num_convolves, num_weights, num_filters)
        # sum out num_weights
        alphaDXW = np.sum(alphaDXW, axis=2) # (num_examples, num_convolves, num_filters)

        # sum and sigmoid
        preds = sigmoid(np.sum(alphaDXW, axis=(1, 2))) # (num_examples)
        residuals = preds - y_true

        # update alpha -- (num_convolves, num_filters)
        # DX -- (num_examples, num_convolves, num_weights, num_filters)

        DXW = np.sum(DX * W, axis=2) # (num_examples, num_convolves, num_filters)
        J = DXW.reshape(DXW.shape[0], -1) # (num_examples, num_convolves * num_filters)

        J = np.expand_dims((preds - y_true) * preds * (1.-preds), 1) * J

        alpha_diff = get_W_diff(y_true, J, residuals, penalty=penalty) # (num_convolves * num_filters)
        alpha_diff = alpha_diff.reshape(num_convolves, num_filters) # (num_convolves, num_filters)

        alpha_next = alpha + alpha_diff

        alpha = .9 * alpha + .1 * alpha_next



        temp_loss += np.sum(np.square(residuals))

    loss.append(temp_loss)
    if temp_loss - min(loss) > 10 and j > np.argmin(loss) + 10:
        break

    
    # if j % 100 == 0:
    #     print((labels[i] == 1)[:50])
    #     print(preds[:50])

plt.plot(loss)
plt.title("Loss with " + str(num_filters) + " filters in Hidden Layer and Sigmoid")
plt.show()

        # print(preds[:50] * y_true[:50])
        # print(preds[:50] * (1 - y_true[:50]))
        # print(np.linalg.norm(residuals))
        # print()
    # if np.linalg.norm(W - temp) < 1e-5:
    #     break
    # if j % 10 == 0:
    #     print((labels[i] == 1)[:50])
    #     print(preds[:50])
    #     print(np.linalg.norm(residuals))





