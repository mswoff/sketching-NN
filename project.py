import numpy as np
import pickle
import scipy
from sklearn.linear_model import Ridge
import time
from helpers import *

data_list, labels = load()
kernel_sz = 3
stride = 2
num_filters = 40
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

alpha = np.random.choice([-1, 1], size=(num_convolves, num_filters))

for j in range(1001):
    # pen = 1.6**(j/10) + 1000*j
    # pen = 1
    
    for i in range(3):
        X = Xs[i][:500]
        y_true = (labels[i][:500] == 1)
        XW = X @ W
        D = XW > 0

        # print(np.sum(D), D.shape[0] * D.shape[1]*D.shape[2])
        # print(D.shape, X.shape, W.shape, np.sum(D))

        DX = np.expand_dims(D, 2) * np.expand_dims(X, 3)    # (num_examples, num_convolves, num_weights, num_filters)
        preds = np.sum(DX * W, axis=2)                      # (num_examples, num_convolves, num_filters)
        preds = preds * alpha                               # (num_examples, num_convolves, num_filters)
        preds = np.sum(preds, axis=(1, 2))                  # (num_examples)
        residuals = preds - y_true

        J = np.sum(DX, axis=1) # (num_examples, num_weights, num_filters)
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

        # weights the positive and negative examples equally
        tot_true = np.sum(y_true)
        weighting =  (y_true.shape[0] - tot_true) / tot_true 
        sample_weights = np.ones(y_true.shape)
        sample_weights[y_true] *= weighting
        # positive examples are 97/3 and negative are 1 if 97 neg and 3 pos examples


        clf = Ridge(alpha=1000000, fit_intercept=True)
        clf.fit(J, residuals, sample_weight=sample_weights) 
        temp = clf.coef_

        W_diff = temp


        print(np.linalg.norm(W_diff))
        print(preds)

        W_diff = W_diff.reshape(kernel_sz**2 * 3, num_filters) # (num_weights, num_filters)

        W_next = W + W_diff
        # if np.linalg.norm(W - temp) < 1e-5:
        #     break
        # print(np.linalg.norm(W - W_next))

        # ne, _, _, _ = np.linalg.lstsq(A, residuals, rcond=None) # use this instead I think
        # a = np.average((W - temp) / ne.reshape(kernel_sz**2 * 3, num_filters))
        # print(np.linalg.norm(W - a * ne.reshape(kernel_sz**2 * 3, num_filters) - temp))
        W = W_next
        # print(preds[:50])
    # if np.linalg.norm(W - temp) < 1e-5:
    #     break
    # if j % 100 == 0:
    #     print((labels[i] == 1)[:50])
    #     print(preds[:50])





