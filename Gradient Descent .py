import numpy as np
from data import features, targets, features_test, targets_test

# define activation function: sigmoid function
def activation(x):
	return 1 / (np.exp(-x) + 1)

n_data, n_features = features.shape


# Initialize weights, Gaussian Normalization
weights = np.random.normal(scale=1/n_features** .5, size=n_features)

# neural network parameters
epochs = 50
learning_rate = 0.01

for e in epochs:
	delta_w = np.zeros(weights.shape)

	for x, y in zip(features, targets):
		output = sigmoid(np.dot(x, weights))
		error = y - output

		# sigmoid function's derivative: f' = f * (1 - f)
		delta_w += error * output * (1 - output) * x

	# use MSE here, so need to divide the number of data
	weights += learning_rate * delta_w / n_data	

	out = sigmoid(np.dot(features, weights))
	loss = np.mean((out - targets) ** 2)
	print(loss)

# accuracy
test_out = sigmoid(np.dot(features_test, weights))
accuracy = np.mean(test_out, targets_test)