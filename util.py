import numpy as np
import pickle

def create_batches(data, batch_length, normalized=True):
	# split each stock data into windows of size batch_length
	m, n = data.shape
	data = data[:, :-(n % batch_length)]
	X = np.reshape(data, (m, n//batch_length, batch_length))

	# subtarct mean and divide by variance
	if normalized:
		mean = X.mean(axis=2, keepdims=True)
		X -= mean

		std = X.std(axis=2, keepdims=True)
		std[std == 0] = 1e-13
		X /= std
	
	# true label is the final price of the next batch
	y = X[:, :, -1]

	# keep dimensions the same
	y = np.reshape(y, (m, n//batch_length, 1))

	# last batch of X will not have a true label, remove it
	X, y = X[:-1], y[1:]
	return X, y