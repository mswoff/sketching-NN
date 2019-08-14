import numpy as np
import pickle
import scipy
from sklearn.linear_model import Ridge
import time
import matplotlib.pyplot as plt 
from helpers import *


data_list, labels = load()
kernel_sz = 3
stride = 2
num_filters = 1
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

loss = [np.inf]

W = np.random.randn(kernel_sz**2 * 3, num_filters) / num_filters / num_weights / X.shape[1]
s_size = 5000
for j in range(1000):
    # pen = 1.6**(j/10) + 1000*j
    # pen = 1
    temp_loss = 0
    diffs = []
    
    for i in range(5):
        X = Xs[i] # (num_examples, num_convolves, num_weights)
        S = np.random.randn(s_size, X.shape[0])
        # S = np.eye(X.shape[0])

        
        y_true = (labels[i] == 1)
        y_true = S @ y_true

        X_compact = np.sum(X, axis=1) # (num_examples, num_weights)
        X_compact = S @ X_compact
        XW = X_compact @ W # (num_examples, num_filters)


        # print(np.sum(D), D.shape[0] * D.shape[1]*D.shape[2])
        # print(D.shape, X.shape, W.shape, np.sum(D))

        preds = sigmoid(np.sum(XW, axis=1)) # (num_examples)
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
        J = X_compact # (num_examples, num_weights)
        # print(J.shape, DX.shape)

        # J = J.reshape(-1, num_weights * num_filters) # (num_examples, num_weights * num_filters) 
        J = np.expand_dims(preds * (1.-preds), 1) * J

        # A = np.concatenate([np.eye(num_weights * num_filters), A])

        # JW_t = J @ W.reshape(-1)
        # b = residuals - JW_t
        # b = (labels[i] == 1)[:100]
        # b = np.concatenate([W.reshape(-1), b])
        # weights = [pen] * num_weights * num_filters + [1] * X.shape[0]
        # W_diff, _, _, _ = np.linalg.lstsq(J, residuals, rcond=None) # (num_weights * num_filters)

        # print(preds[:50])

        # weights the positive and negative examples equally
        # tot_true = np.sum(y_true)
        # weighting =  (y_true.shape[0] - tot_true) / tot_true 
        # sample_weights = np.ones(y_true.shape)
        # sample_weights[y_true] *= weighting
        # positive examples are 97/3 and negative are 1 if 97 neg and 3 pos examples


        clf = Ridge(alpha=100000, fit_intercept=True)
        clf.fit(J, -residuals) 
        temp = clf.coef_

        W_diff = temp


        print(np.linalg.norm(W_diff), j)
        diffs.append(np.linalg.norm(W_diff))

        W_diff = W_diff.reshape(kernel_sz**2 * 3, num_filters) # (num_weights, num_filters)

        W_next = W + W_diff
        # if np.linalg.norm(W - temp) < 1e-5:
        #     break
        # print(np.linalg.norm(W - W_next))

        # ne, _, _, _ = np.linalg.lstsq(A, residuals, rcond=None) # use this instead I think
        # a = np.average((W - temp) / ne.reshape(kernel_sz**2 * 3, num_filters))
        # print(np.linalg.norm(W - a * ne.reshape(kernel_sz**2 * 3, num_filters) - temp))
        W = W * .9 + .1 * W_next
        # print(preds[:50])
    # if np.linalg.norm(W - temp) < 1e-5:
    #     break

        # make predictions and evaluate loss
        X = Xs[i] # (num_examples, num_convolves, num_weights)
        
        y_true = (labels[i] == 1)

        X_compact = np.sum(X, axis=1) # (num_examples, num_weights)
        XW = X_compact @ W # (num_examples, num_filters)real_residuals = real_preds - (labels[i] == 1)
        preds = sigmoid(np.sum(XW, axis=1))
        real_residuals = preds - y_true

        temp_loss += np.sum(np.square(real_residuals))
    print(temp_loss)
    print()

    if temp_loss - min(loss) > .0000001 and j > np.argmin(loss) + 10:
        break




    loss.append(temp_loss)
    # if j % 100 == 0:
    #     print((labels[i] == 1)[:50])
    #     print(preds[:50])

plt.plot(loss[1:])
plt.title("Loss with Gaussian Sketch(" + str(s_size) + ") and no Hidden Layer")
plt.show()


