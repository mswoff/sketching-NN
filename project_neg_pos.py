import numpy as np
import pickle
import scipy
from sklearn.linear_model import Ridge
import time

import matplotlib.pyplot as plt 
from helpers import *

def get_W_diff(y_true, J, residuals, penalty=10000):
    # weights the positive and negative examples equally
    # tot_true = np.sum(y_true)
    # weighting =  (y_true.shape[0] - tot_true) / tot_true 
    # sample_weights = np.ones(y_true.shape)
    # sample_weights[y_true] *= weighting
    # positive examples are 97/3 and negative are 1 if 97 neg and 3 pos examples


    clf = Ridge(alpha=penalty, fit_intercept=False)
    # clf.fit(J, residuals, sample_weight=sample_weights) 
    clf.fit(J, residuals)
    diff = clf.coef_
    return diff

data_list, labels = load()
kernel_sz = 3
stride = 2
num_filters = 10
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

MSEs = []
accs = []

for h in range(2):
    W = np.random.randn(kernel_sz**2 * 3, num_filters) / num_filters / num_weights / X.shape[1] / 100

    num_convolves = X.shape[1]
    # alpha = np.random.choice([-1, 1], size=(num_convolves, num_filters))
    alpha = np.random.randn(num_convolves, num_filters) / num_filters / num_weights / X.shape[1] / 100

    MSE_loss = []
    pred_accuracy = []

    for j in range(201):
        penalty = 100 + j
        temp_loss = 0
        num_per_batch = 10000
        s_size = 200
        gaussian_sample = True


        thresh = .5
        tot_correct = 0
        tot_seen = 0
        loss = 0
        
        for i in range(1):
            X = Xs[i][:num_per_batch]     # (num_examples, num_convolves, num_weights)
            y_true = (labels[i] == 1)[:num_per_batch]
            XW = X @ W          # (num_examples, num_convolves, num_filters)
            D = XW > 0          # (num_examples, num_convolves, num_filters)
            aD = alpha * D      # (num_examples, num_convolves, num_filters)

            # DX = np.expand_dims(D, 2) * np.expand_dims(X, 3) # (num_examples, num_convolves, num_weights, num_filters)
            # # multiply by alpha
            # alphaDX = np.expand_dims(alpha, 1) * DX # (num_examples, num_convolves, num_weights, num_filters)
            # # multiply by weights
            # alphaDXW = alphaDX * W # (num_examples, num_convolves, num_weights, num_filters)
            # # sum out num_weights
            # alphaDXW = np.sum(alphaDXW, axis=2) # (num_examples, num_convolves, num_filters)

            # sum and sigmoid
            preds = sigmoid(np.sum(aD * XW, axis=(1, 2))) # (num_examples)
            residuals = preds - y_true

            # Jacobian
            aDX = np.expand_dims(aD, 2) * np.expand_dims(X, 3)    # (num_examples, num_convolves, num_weights, num_filters)
            J = np.sum(aDX, axis=1)                      # (num_examples, num_weights, num_filters)
            J = J.reshape(-1, num_weights * num_filters) # (num_examples, num_weights * num_filters)
            J = np.expand_dims(preds * (1.-preds), 1) * J # (num_examples, num_weights * num_filters) 


            if gaussian_sample:
                S = np.random.randn(s_size, num_per_batch)
                residuals_sketch = S @ residuals
                J_sketch = S @ J

                W_diff = get_W_diff(y_true, J_sketch, -residuals_sketch, penalty=penalty)
            else:
                W_diff = get_W_diff(y_true, J, -residuals, penalty=penalty)

            W_diff = W_diff.reshape(kernel_sz**2 * 3, num_filters) # (num_weights, num_filters)

            W_next = W + W_diff
            W = .9 * W + .1 * W_next



            ################################
        #     XW = X @ W
        #     D = XW > 0

        #     # print(np.sum(D), D.shape[0] * D.shape[1]*D.shape[2])
        #     # print(D.shape, X.shape, W.shape, np.sum(D))

        #     DX = np.expand_dims(D, 2) * np.expand_dims(X, 3) # (num_examples, num_convolves, num_weights, num_filters)
        #     # multiply by alpha
        #     alphaDX = np.expand_dims(alpha, 1) * DX # (num_examples, num_convolves, num_weights, num_filters)
        #     # multiply by weights
        #     alphaDXW = alphaDX * W # (num_examples, num_convolves, num_weights, num_filters)
        #     # sum out num_weights
        #     alphaDXW = np.sum(alphaDXW, axis=2) # (num_examples, num_convolves, num_filters)

        #     # sum and sigmoid
        #     preds = sigmoid(np.sum(alphaDXW, axis=(1, 2))) # (num_examples)
        #     residuals = preds - y_true

            # # update alpha -- (num_convolves, num_filters)
            # # DX -- (num_examples, num_convolves, num_weights, num_filters)

            XW = X @ W # (num_examples, num_convolves, num_filters)
            D = XW > 0 # (num_examples, num_convolves, num_filters)

            DXW = D * XW # (num_examples, num_convolves, num_filters)

            preds = sigmoid(np.sum(alpha * DXW, axis=(1, 2))) # (num_examples)
            
            residuals = preds - y_true

            J = DXW.reshape(DXW.shape[0], -1) # (num_examples, num_convolves * num_filters)
            J = np.expand_dims(preds * (1.-preds), 1) * J

            clf = Ridge(alpha=penalty, fit_intercept=False) # sample_weight=sample_weights

            if gaussian_sample:
                S = np.random.randn(s_size, num_per_batch)
                residuals_sketch = S @ residuals
                J_sketch = S @ J

                clf.fit(J_sketch, -residuals_sketch)
            else:
                clf.fit(J, -residuals)

            alpha_diff = clf.coef_

            alpha_diff = alpha_diff.reshape(num_convolves, num_filters) # (num_convolves, num_filters)

            alpha_next = alpha + alpha_diff

            alpha = .9 * alpha + .1 * alpha_next

            # alpha = .9 * alpha + .1 * alpha_next


            square_res = np.square(residuals)
            probs = square_res / np.sum(square_res)

            tot_correct += num_per_batch - np.sum(np.logical_xor(preds > thresh, y_true))
            tot_seen += num_per_batch

            loss = square_res
            MSE_loss.append(loss / s_size * tot_seen/num_per_batch)
            pred_accuracy.append(tot_correct / tot_seen)

            tot_tru_correct = np.sum(np.logical_and(preds > thresh, y_true))
            tot_true = np.sum(preds > thresh)
            tot_true_total = np.sum(y_true)

            print(pred_accuracy[-1], tot_true , tot_tru_correct, tot_true_total)

        #     temp_loss += np.sum(np.square(residuals))

        # loss.append(temp_loss)
        # if temp_loss - min(loss) > 10 and j > np.argmin(loss) + 10:
        #     break



    MSEs.append(MSE_loss)
    accs.append(pred_accuracy)
    print(np.average(pred_accuracy[-2:]))

# print(alpha)
# print(np.sum(alpha))
# print(alpha.shape)
# print(np.sum(alpha > 0), np.sum(alpha < 0))
for MSE_loss in MSEs:
    plt.plot(MSE_loss)
plt.title("MSE on " + str(num_per_batch) + " examples with " + str(s_size) + " dim random Guassian sketch, " + str(num_filters) + " filters, reweighted relu activations with Sigmoid")
# plt.legend()
plt.show()

for pred_accuracy in accs:
    plt.plot(pred_accuracy)
# plt.legend()
plt.title("Accuracies on " + str(num_per_batch) + " examples with " + str(s_size) + " dim random Gaussian sketch, " + str(num_filters) + " filters, reweighted relu activations with Sigmoid")
plt.show()





