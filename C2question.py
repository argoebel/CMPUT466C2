# -*- coding: utf-8 -*-
import numpy as np
import struct
import matplotlib.pyplot as plt
from scipy.special import expit
import sys


def readMNISTdata():

    with open('t10k-images-idx3-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        test_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_data = test_data.reshape((size, nrows*ncols))

    with open('t10k-labels-idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        test_labels = np.fromfile(
            f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_labels = test_labels.reshape((size, 1))

    with open('train-images-idx3-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        train_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_data = train_data.reshape((size, nrows*ncols))

    with open('train-labels-idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        train_labels = np.fromfile(
            f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_labels = train_labels.reshape((size, 1))

    # augmenting a constant feature of 1 (absorbing the bias term)
    train_data = np.concatenate(
        (np.ones([train_data.shape[0], 1]), train_data), axis=1)
    test_data = np.concatenate(
        (np.ones([test_data.shape[0], 1]),  test_data), axis=1)
    np.random.seed(314)
    np.random.shuffle(train_labels)
    np.random.seed(314)
    np.random.shuffle(train_data)

    X_train = train_data[:50000] / 256
    t_train = train_labels[:50000]

    X_val = train_data[50000:] / 256
    t_val = train_labels[50000:]

    return X_train, t_train, X_val, t_val, test_data, test_labels


def predict(X, W, t=None):
    # X_new: Nsample x (d+1)
    # W: (d+1) x K
    # t: Nsample x 1

    # TODO Your code here

    # z: Nsample x K
    z = X.dot(W)

    # https://stackoverflow.com/questions/16202348/numpy-divide-row-by-row-sum
    # subtract max of each row from each row item
    z = z - z.max(axis=1)[:, None]

    # y = Xw
    y = np.exp(z)

    # https://stackoverflow.com/questions/16202348/numpy-divide-row-by-row-sum
    # Divide row by its sum

    y = y/y.sum(axis=1)[:, None]

    # return index of highest value in row of y
    t_hat = np.argmax(y, axis=1)

    # print(y)

    # loss calculated by max of each y row and averaged across all values
    loss = 0
    for sample in range(X.shape[0]):
        loss += np.sum(t[sample]*np.log2(max(y[sample])))

    loss = -loss / X.shape[0]

    # calculate accuracy by counting the number of correct predictions
    count = 0
    for sample in range(X.shape[0]):
        if t_hat[sample] == t[sample]:
            count += 1

    acc = count / X.shape[0]

    t_hat = np.reshape(t_hat, (-1, 1))

    return y, t_hat, loss, acc


def train(X_train, y_train, X_val, t_val):
    N_train = X_train.shape[0]
    N_val = X_val.shape[0]

    # TODO Your code here

    w = np.zeros([X_train.shape[1], 10])

    epoch_best = 0
    acc_best = 0
    W_best = None

    for epoch in range(MaxIter):
        loss_this_epoch = 0
        for b in range(int(np.ceil(N_train/batch_size))):
            X_batch = X_train[b*batch_size: (b+1)*batch_size]
            t_batch = y_train[b*batch_size: (b+1)*batch_size]

            y, t_hat, loss, acc = predict(X_batch, w, t_batch)
            loss_this_epoch += loss

            # print("y")
            # print(y)
            # print("t_hat")
            # print(t_hat)
            # print("t_batch")
            # print(t_batch)

            featureMatrix = np.zeros([y.shape[0], y.shape[1]])
            testMatrix = np.zeros([y.shape[0], y.shape[1]])

            for item in range(len(t_batch)-1):
                featureMatrix[item][t_batch[item]] = 1

            for item in range(len(t_batch)-1):
                testMatrix[item][t_hat[item]] = 1
            # print("Alpha: ", alpha)
            w = w - alpha * \
                (X_batch.T.dot(testMatrix - featureMatrix) + decay * w)

        lossArray.append(loss_this_epoch/int(np.ceil(N_train/batch_size)))

        # Validation
        y, t_hat, loss, acc = predict(X_val, w, t_val)
        valArray.append(acc)

        if acc > acc_best:
            acc_best = acc
            epoch_best = epoch
            W_best = w

            # print("BEST")
            # print("w")
            # print(W_best)
            # print(loss)
            # print(acc)

    return epoch_best, acc_best,  W_best


##############################
# Main code starts here
X_train, t_train, X_val, t_val, X_test, t_test = readMNISTdata()


print(X_train.shape, t_train.shape, X_val.shape,
      t_val.shape, X_test.shape, t_test.shape)


N_class = 10

alpha = 0.05      # learning rate
batch_size = 100    # batch size
MaxIter = 50        # Maximum iteration
decay = 0.          # weight decay
maxAlpha = 1

epoch_best = 0
acc_best = 0
acc_test_best = 0
best_alpha = 0
best_loss = []
best_val = []
W_best = None

alphaArray = []
accuracyArray = []

totalLossArray = []
totalValArray = []

for i in np.arange(alpha, maxAlpha, 0.05):
    print("Current Alpha: ", i)
    alpha = i
    lossArray = []
    valArray = []
    epoch, acc, W = train(X_train, t_train, X_val, t_val)

    if acc > acc_best:
        acc_best = acc
        best_alpha = i
        epoch_best = epoch
        W_best = W
        best_loss = lossArray
        best_val = valArray

    alphaArray.append(i)
    accuracyArray.append(acc)
    totalLossArray.append(lossArray)
    totalValArray.append(valArray)

_, _, _, acc_test_best = predict(X_test, W_best, t_test)


# print('At epoch', epoch_best, 'val: ', acc_best,
#       'test:', acc_test, 'train:', acc_train)
# print('At epoch', epoch_best, 'val: ', acc_best,
#       'test:', acc_test)

print("Best alpha: ", best_alpha)
print("Best validation accuracy: ", acc_best)
print("Best test accuracy: ", acc_test_best)
print("Best loss: ", best_loss)
print("Accuracy array: ")
print(accuracyArray)


x_axis1 = np.arange(0, MaxIter, 1)
x_axis2 = np.arange(0.05, maxAlpha, 0.05)
y_axis1 = np.arange(0, 1, 0.01)
y_axis2 = np.arange(0, 1, 0.01)

plt.subplot(2, 2, 3)
plt.plot(x_axis1, best_loss)
plt.grid()
# plt.ylim(0, 0.2)
plt.ylabel("Loss")
plt.xlabel("Epoch")


plt.subplot(2, 2, 1)
plt.plot(x_axis1, best_val)
plt.grid()
# plt.ylim(0, 1)
plt.title("Loss and Accuracy Over Epochs")
plt.ylabel("Accuracy")

plt.subplot(2, 2, 2)
plt.plot(x_axis2, accuracyArray)
plt.grid()
plt.title("Accuracy Over Alpha Values")
# plt.ylim(0, 1)
# plt.title("Risk Over Epochs")
# plt.ylabel("Accuracy")
plt.xlabel("Alpha")

plt.savefig('C2Q2')

plt.show()
