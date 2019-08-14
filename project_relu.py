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
    W = np.random.randn(kernel_sz**2 * 3, num_filters) / num_filters / num_weights / X.shape[1]

    num_convolves = X.shape[1]

    # alpha = np.random.choice([-1, 1], size=(num_convolves, num_filters))
    alpha = np.random.randn(num_convolves, num_filters) / num_filters / num_weights / X.shape[1]

    a_pos = alpha > 0

    MSE_loss = []
    pred_accuracy = []
    num_per_batch = 10000
    s_size = 200
    random_sample = True
    gaussian_sample = not random_sample

    probs = np.ones(num_per_batch) / num_per_batch

    thresh = 0


    for j in range(151):
        pen = 100 + j
        tot_correct = 0
        tot_seen = 0
        loss = 0

        for i in range(1):
            X = Xs[i][:num_per_batch] # (num_examples, num_convolves, num_weights)

            y_true_bool = (labels[i][:num_per_batch] == 1) 
            y = y_true_bool * 1. - (labels[i][:num_per_batch] != 1) * 1.

            if random_sample:
                S = np.random.choice(np.arange(num_per_batch), size=s_size, p=probs)

                # sketch
                X = X[S]
                y_true_bool = y_true_bool[S]
                y = y[S]

            XW = X @ W # (num_examples, num_convolves, num_filters)
            D = XW > 0

            aD = alpha * D # (num_examples, num_convolves, num_filters)

            preds = np.sum(aD * XW , axis=(1, 2)) # (num_examples)
            
            residuals = preds - y
            
            # keep track of loss anc accuracy metrics
            loss += np.sum(np.square(residuals))

            # Jacobian
            aDX = np.expand_dims(aD, 2) * np.expand_dims(X, 3)    # (num_examples, num_convolves, num_weights, num_filters)
            J = np.sum(aDX, axis=1)                     # (num_examples, num_weights, num_filters)
            J = J.reshape(-1, num_weights * num_filters) # (num_examples, num_weights * num_filters) 

            # Gaussian sampling
            if gaussian_sample:
                S = np.random.randn(s_size, num_per_batch)
                residuals = S @ residuals
                J = S @ J


            # weights the positive and negative examples equally

            # positive examples are 97/3 and negative are 1 if 97 neg and 3 pos examples


            clf = Ridge(alpha=pen, fit_intercept=False) # sample_weight=sample_weights
            clf.fit(J, -residuals) 
            W_diff = clf.coef_

            # print(np.linalg.norm(W_diff), np.linalg.norm(residuals), i, j)

            W_diff = W_diff.reshape(kernel_sz**2 * 3, num_filters) # (num_weights, num_filters)

            W_next = W + W_diff
            # if np.linalg.norm(W - temp) < 1e-5:
            #     break
            # print(np.linalg.norm(W - W_next))

            W = .9 * W + .1 * W_next

            # update alpha -- (num_convolves, num_filters)
            # DX -- (num_examples, num_convolves, num_weights, num_filters)

            XW = X @ W # (num_examples, num_convolves, num_filters)
            D = XW > 0 # (num_examples, num_convolves, num_filters)

            DXW = D * XW # (num_examples, num_convolves, num_filters)
            J = DXW.reshape(DXW.shape[0], -1) # (num_examples, num_convolves * num_filters)

            preds = np.sum(alpha * DXW, axis=(1, 2)) # (num_examples)
            
            residuals = preds - y

            if gaussian_sample:
                S = np.random.randn(s_size, num_per_batch)
                residuals = S @ residuals
                J = S @ J

            clf = Ridge(alpha=pen, fit_intercept=False) # sample_weight=sample_weights
            clf.fit(J, -residuals) 
            alpha_diff = clf.coef_

            alpha_diff = alpha_diff.reshape(num_convolves, num_filters) # (num_convolves, num_filters)

            alpha_next = alpha + alpha_diff

            alpha = .9 * alpha + .1 * alpha_next



            # calculate accuracy and probabilities for next iteration
            # if j % 5 == 0:
            X = Xs[i][:num_per_batch]
            y_true_bool = (labels[i][:num_per_batch] == 1) 

            XW = X @ W
            D = XW > 0
            aD = alpha * D # (num_examples, num_convolves, num_filters)

            preds = np.sum(aD * XW, axis=(1, 2))               # (num_examples)

            y = y_true_bool * 1. - (labels[i][:num_per_batch] != 1) * 1.
            residuals = preds - y
            # sample according to squared residuals
            square_res = np.square(residuals)
            probs = square_res / np.sum(square_res)

            tot_correct += num_per_batch - np.sum(np.logical_xor(preds > thresh, y_true_bool))
            tot_seen += num_per_batch

            MSE_loss.append(loss / s_size * tot_seen/num_per_batch)
            pred_accuracy.append(tot_correct / tot_seen)

            tot_tru_correct = np.sum(np.logical_and(preds > thresh, y_true_bool))
            tot_true = np.sum(preds > thresh)
            tot_true_total = np.sum(y_true_bool)

            print(pred_accuracy[-1], tot_true , tot_tru_correct, tot_true_total)

            # if j % 10  == 0:
            #     print(j)
            #     print(preds[:50] * (labels[i][:50] == 1))
            #     print(preds[:50] * (labels[i][:50] != 1))




    MSEs.append(MSE_loss)
    accs.append(pred_accuracy)
    print(np.average(pred_accuracy[-2:]))

# print(alpha)
# print(np.sum(alpha))
# print(alpha.shape)
# print(np.sum(alpha > 0), np.sum(alpha < 0))
# print(np.sum(np.logical_xor(a_pos, alpha > 0)))
for MSE_loss in MSEs:
    plt.plot(MSE_loss)
plt.title("MSE on " + str(num_per_batch) + " examples with " + str(s_size) + " dim random Guassian sketch, " + str(num_filters) + " filters, reweighted relu activations")
# plt.legend()
plt.show()

for pred_accuracy in accs:
    plt.plot(pred_accuracy)
# plt.legend()
plt.title("Accuracies on " + str(num_per_batch) + " examples with " + str(s_size) + " dim random Gaussian sketch, " + str(num_filters) + " filters, reweighted relu activations")
plt.show()





