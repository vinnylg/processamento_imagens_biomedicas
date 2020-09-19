import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn import metrics
from sklearn import svm
import statistics

from extraction import getLBP, getTopHat

# dataset = [
#     {
#         'patient_id': 'number for patient identification on dataset',
#         'data': 'list of all images features, data[i] can be list of histogram, means, etc',
#         'target': 'target/label for classification of data[i] feature'
#     }
#     ... array of patients
#     ]

def runSVM(dataset):
	start_time = time.time()
	preds = []
	values = []
	#prepare model
	model = svm.SVC(kernel='rbf', random_state=0, gamma=1,C=1, decision_function_shape='ovo')

	#for each patient
	for index in range(len(dataset)):
		#split dataset into validation set and training set
		#leave one out
		train = dataset[:index] + dataset[index+1:]
		test = [dataset[index]]
		#k-fode simples
		# train = dataset[20:]
		# test = dataset[:20]

		#prepare data for training
		trainX = []
		trainY = []
		for patient in train:
			for data, target in zip(patient['data'], patient['target']):
				trainX.append(data)
				trainY.append(target)

		trainX = np.array(trainX,dtype='object')
		trainX = normalize(trainX)

		trainY = np.array(trainY,dtype='float')

		print(f"{trainX.shape} features to fit from {len(train)} patients, without patient {index}")

		#fit model
		model.fit(trainX, trainY)

		#prepare validation data
		testX = []
		testY = []
		for patient in test:
			for data, target in zip(patient['data'], patient['target']):
				testX.append(data)
				testY.append(target)

		testX = np.array(testX, dtype='object')
		testX = normalize(testX)

		testY = np.array(testY, dtype='float')

		print(f"{testX.shape} features to test for patient {index}")

		#get predictions for each instance in validation data
		pred = model.predict(testX)

		preds.append(pred)
		values.append(testY)
		print(f"accuracy: {metrics.accuracy_score(testY, pred)}")
		print(f" Time elapsed: {datetime.timedelta(seconds=round(time.time()-start_time))} seconds")

	return preds,values

def main():
	if(not os.path.isdir('./predictions')):
		os.mkdir('./predictions')

	preds,values = runSVM(getTopHat())

	with open('./predictions/svm_tophat.npz', 'wb') as out:
		np.savez(out, preds=preds,values=values)
		print(f"./predictions/svm_tophat.npz saved")

	# preds, values = runSVM(getLBP())
	# with open('./predictions/svm_LBP.npz', 'wb') as out:
	# 	np.savez(out, preds=preds, values=values)
	# 	print(f"./predictions/svm_LBP.npz saved")


if __name__ == "__main__":
	main()
