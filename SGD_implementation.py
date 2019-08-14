import numpy as np
import pickle
import scipy
from sklearn.linear_model import Ridge
import time
import matplotlib.pyplot as plt

import tensorflow as tf

import keras
from keras.models import Model
from keras.layers import Input, Conv2D, Flatten, Dense, Dot

from helpers import *


Xs, labels = load()
kernel_sz = 3
stride = 2
num_filters = 41
num_weights = kernel_sz**2 * 3

temp = []
for i in range(0, 32-kernel_sz+1, stride):
    for j in range(0, 32-kernel_sz+1, stride):
        d = Xs[0][:, i:i+kernel_sz, j:j+kernel_sz, :].reshape(-1, num_weights)
        temp.append(d)
temp = np.array(temp).transpose([1, 0, 2])
_, num_convs, num_weights = temp.shape

MSEs = []
accs = []

for h in range(2):
    MSE_loss = []
    pred_accuracy = []
    num_per_batch = 100


    thresh = 0

    X = Xs[0][:num_per_batch] # (num_examples, num_convolves, num_weights)

    y_true_bool = (labels[0][:num_per_batch] == 1) 
    y = y_true_bool * 1. #- (labels[0][:num_per_batch] != 1) * 1.

    inputs = Input(shape=(32, 32, 3))

    x = Conv2D(filters=num_filters, kernel_size=kernel_sz, strides=stride, padding='valid',  # keras.initializers.RandomNormal(mean=0.0, stddev=num_filters * num_weights * X.shape[1])
        kernel_initializer="he_normal",
        use_bias=False, activation=tf.nn.relu)(inputs)
    x = Flatten(name="flat")(x)

    preds = Dense(1, activation="sigmoid", use_bias=False)(x)
    # preds = Dot(1, name="out")([x, alphas])



    model = Model(inputs=inputs, outputs=preds)
    model.compile(optimizer=keras.optimizers.SGD(lr=0.001),
              loss='mean_squared_error',
              metrics=['accuracy'])

    alpha = np.random.choice([-1., 1], size=(num_convs * num_filters))

    model.fit(X, y, epochs=1000, batch_size=100)


    preds = model.predict([X, alpha])
    print(preds[:100])
    
    residuals = preds - y
    
    # keep track of loss anc accuracy metrics
    loss = np.sum(np.square(residuals))




    tot_correct = num_per_batch - np.sum(np.logical_xor(preds > thresh, y_true_bool))
    tot_seen = num_per_batch

    # MSE_loss.append(loss / s_size * tot_seen/num_per_batch)
    # pred_accuracy.append(tot_correct / tot_seen)

    tot_tru_correct = np.sum(np.logical_and(preds > thresh, y_true_bool))
    tot_true = np.sum(preds > thresh)
    tot_true_total = np.sum(y_true_bool)

    print(pred_accuracy[-1], tot_true , tot_tru_correct, tot_true_total)






# for MSE_loss in MSEs:
#     plt.plot(MSE_loss)
# plt.title("MSE on " + str(num_per_batch) + " examples with " + str(s_size) + " dim random sample sketch, " + str(num_filters) + " filters, relu activations")
# # plt.legend()
# plt.show()

# for pred_accuracy in accs:
#     plt.plot(pred_accuracy)
# # plt.legend()
# plt.title("Accuracies on " + str(num_per_batch) + " examples with " + str(s_size) + " dim random sample sketch, " + str(num_filters) + " filters, relu activations")
# plt.show()
