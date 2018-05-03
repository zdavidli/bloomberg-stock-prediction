import numpy as np
import pickle

def create_batches(data, batch_length, normalized=True):
	# split each stock data into windows of size batch_length
	m, n = data.shape
	data = data[:, :-(n % batch_length)]
	# print(data.shape)
	# print(m)
	# print(n)
	X = np.reshape(data, (m, n//batch_length, batch_length))
	# print(X.shape)
	

	# subtarct mean and divide by variance
	mean = None
	if normalized:
		# subtract mean
		mean = X.mean(axis=2, keepdims=True)
		X -= mean
		# print(X.shape)

		# divide by standard dev
		# std = X.std(axis=2, keepdims=True)
		# std = np.repeat(std, batch_length, axis=2)
		# print(std.shape)
		# avoid divide by zero
		# std[std == 0] = 1e-13
		# X /= std
		# print(X.shape)


	# true label is the final price of the next batch
	y = X[:, 1:, -1]
	# mean = None
	if normalized:
		# subtract mean
		# mean = X.mean(axis=2, keepdims=True)
		# X -= mean
		y -= mean[:,:-1,0]
	

	# keep dimensions the same
	y = np.reshape(y, (m, n//batch_length - 1, 1))

	# last batch of X will not have a true label, remove it
	X, y = X[:,:-1], y[:,:]
	return X, y

def sliding_window(data, batch_length, overlap):
	m, n = data.shape
	windows = n // (batch_length - overlap) - batch_length//overlap
	X = np.ndarray((m, windows, batch_length))
	y = np.ndarray((m, windows, 1))
	for i in range(windows):
		X[:,i,:] = data[:,i*batch_length-i*overlap:i*batch_length-i*overlap+batch_length]
		y[:,i,0] = data[:,i*batch_length-i*overlap + batch_length-overlap]
		# print(X[:,i,:])
		# print(y[:,i,0])
		me = X[:,i,:].mean()
		# std = X[:,i,:].std()
		# print(y[:,i,0])
		X[:,i,:] -= me
		y[:,i,0] -= me
		# X[:,i,:] /= std
		# y[:,i,0] /= std
	# if normalized:
	# 	# subtract mean
	# 	mean = X.mean(axis=2, keepdims=True)
	# 	X -= mean
	# 	# print(X.shape)

	# 	# divide by standard dev
	# 	std = X.std(axis=2, keepdims=True)
	# 	# std = np.repeat(std, batch_length, axis=2)
	# 	# print(std.shape)
	# 	# avoid divide by zero
	# 	std[std == 0] = 1e-13
	# 	X /= std
		# print(X.shape)
	return X, y

