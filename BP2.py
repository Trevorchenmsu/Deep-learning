import numpy as np
import random
from data import inputs, targets

'''
source: https://blog.csdn.net/loseinvain/article/details/78092613
https://blog.csdn.net/LoseInVain/article/details/78239068
''' 

# two hidden layers, one has 3 neurons, one has 2 neurons
n_input = 3
w1_hidden = 3
w2_hidden = 2
n_output = 1

weights = {
	'w1': np.random.normal(size=(n_input, w1_hidden)),
	'w2': np.random.normal(size=(n_input, w2_hidden)),
	'output': np.random.normal(size=(w2_hidden, n_output))
}

biases = {
    'B1': np.random.normal(size=(w1_hidden)),
    'B2': np.random.normal(size=(w2_hidden)),
    'output': np.random.normal(size=(n_output))
}


def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
	f = sigmoid(x)
	return f * (1 - f)

''' forward propagation '''
def forward(x_input):
    h1 = np.dot(x_input, weights['W1']) + biases['B1']
    h1 = sigmoid(h1)

    h2 = np.dot(h1, weights['W2']) + biases['B2']
    h2 = sigmoid(h2)

    output = np.dot(h2, weights['output']) + biases['output']
    output = sigmoid(output)
    return output

def forward_1(x_input):
    z1 = np.dot(x_input, weights['W1'])+biases['B1']
    a1 = sigmoid(z1)
    return z1, a1

def forward_2(x_input):
    z2 = np.dot(x_input, weights['W2'])+biases['B2']
    a2 = sigmoid(z2)
    return z2, a2

def forward_3(x_input):
    z3 = np.dot(x_input, weights['output'])+biases['output']
    a3 = sigmoid(z3)
    return z3, a3

''' backward propagation '''
def cal_gradient(batch, label):
    z1, a1 = forward_1(batch)
    z2, a2 = forward_2(a1)
    z3, a3 = forward_3(a2)

    sig31 = np.sum((a3 - label) * d_sigmoid(z3)) / batch_size
    sig21 = sig31 * weights['output'][0, 0] * np.sum(d_sigmoid(z2[:, 0])) / batch_size
    sig22 = sig31 * weights['output'][1, 0] * np.sum(d_sigmoid(z2[:, 1])) / batch_size

    sig11 = (sig21 * weights['W2'][0, 0] + sig22 * weights['W2'][0, 1]) * np.sum(d_sigmoid(z1[:, 0])) / batch_size
    sig12 = (sig21 * weights['W2'][1, 0] + sig22 * weights['W2'][1, 1]) * np.sum(d_sigmoid(z1[:, 0])) / batch_size
    sig13 = (sig21 * weights['W2'][2, 0] + sig22 * weights['W2'][2, 1]) * np.sum(d_sigmoid(z1[:, 0]))/ batch_size

    sig1 = np.array([sig11, sig12, sig13])
    sig2 = np.array([sig21, sig22])
    sig3 = np.array([sig31])
    return sig3, sig2, sig1


learning_rate = 0.1
def update(batch, sig1, sig2, sig3):
    z1, a1 = forward_1(batch)
    z2, a2 = forward_2(a1)
    z3, a3 = forward_3(a2)
    a1 = np.sum(a1, axis=0) / batch_size
    a2 = np.sum(a2, axis=0) / batch_size
    a3 = np.sum(a3, axis=0) / batch_size
    a1 = np.reshape(a1, (a1.shape[0], 1))
    a2 = np.reshape(a2, (a2.shape[0], 1))
    a3 = np.reshape(a3, (a3.shape[0], 1))
    sig1 = np.reshape(sig1, (sig1.shape[0], 1))
    sig2 = np.reshape(sig2, (sig2.shape[0], 1))
    sig3 = np.reshape(sig3, (sig3.shape[0], 1))

    input = np.sum(batch, axis=0) / batch_size
    weights['W1'] += -learning_rate * (input * np.transpose(sig1))
    weights['W2'] += -learning_rate * (a1 * np.transpose(sig2))
    weights['output'] += -learning_rate * (a2 * np.transpose(sig3))

    biases['B1'] += -learning_rate * np.transpose(sig1)
    biases['B2'] += -learning_rate * np.transpose(sig2)
    biases['output'] += -learning_rate * np.transpose(sig3)


sum = 0
i = 0
import time
begin = time.clock()
while True:
    batch, label = random_get_data(mat, batch_size)
    sig3, sig2, sig1 = cal_gradient(batch, label)
    update(batch, sig1, sig2, sig3)
    loss = cal_loss(batch, label)
    sum += loss
    i += 1
    if i % 1000 == 0:
        loss = sum/1000
        sum = 0
        print(loss)
        if loss < 0.01:
            break
end = time.clock()
print('cost time = %f s' % (end-begin))