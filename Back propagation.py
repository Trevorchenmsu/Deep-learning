import numpy as np
from data import features, targets, features_test, targets_test

def sigmoid(x):
	return 1 / (np.exp(-x)  + 1)

# neurons in a hidden layer
n_hidden = 2
epochs = 100
learn_rate = 0.01

n_records, n_features = features.shape
last_loss = None

# initialize weights
weights_input_hidden = np.random.normal(scale=1 / n_features ** 0.5, size=(n_features, n_hidden))
weights_output_hidden = np.random.normal(scale=1 / n_features ** 0.5, size=(n_hidden))

for e in range(epochs):
	delta_w_input_hidden = np.zeros(weights_input_hidden.shape)
	delta_w_output_hidden = np.zeros(weights_output_hidden.shape)

	for x, y in zip(features, targets):
		# forward propagation
		hiddne_input = np.dot(x, weights_input_hidden)
		hidden_output = sigmoid(hiddne_input)

		output = sigmoid(np.dot(hidden_output, 
								weights_output_hidden))

		# backword propagation
		error = y - output

		output_error = error * output * (1 - output)

		# propagate errors to hidden layer
		hidden_error = np.dot(output_error, weights_input_hidden) * hidden_output * (1 - hidden_output)

		# update weight changes
		delta_w_output_hidden += output_error * hidden_output
		delta_w_input_hidden += hidden_error * x

	# update weights
	weights_input_hidden += learn_rate * delta_w_input_hidden / n_records
	weights_output_hidden += learn_rate * delta_w_output_hidden / n_records

	if e % (epochs / 10) == 0:
		hidden_output = sigmoid(np.dot(x, weights_input_hidden))
		output = sigmoid(np.dot(hidden_output, weights_output_hidden))

		loss = np.mean((out - targets) ** 2)
		print("training loss: ", loss)
		last_loss = loss


# test data
hidden = sigmoid(np.dot(features_test, weights_input_hidden))
output = sigmoid(np.dot(hidden, weights_output_hidden))
predictions = output > 0.5
accuracy = np.mean(predictions == targets_test)