import sys
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
	#prepare model
	model = svm.SVC(kernel='rbf', random_state=0, gamma=1,C=1, decision_function_shape='ovo')

	#for each patient
	for index in range(len(dataset)):
		#split dataset into validation set and training set
		#leave one out
		train = dataset[:index] + dataset[index+1:]
		test = [dataset[index]]

		#prepare data for training
		trainX = []
		trainY = []
		for patient in train:
			for data, target in zip(patient['data'], patient['target']):
				trainX.append(data)
				trainY.append(target)

		print(f"{len(trainX)} features to fit from {len(train)} patients")
		#fit model
		model.fit(trainX, trainY)

		#prepare validation data
		testX = []
		testY = []
		for patient in test:
			for data, target in zip(patient['data'], patient['target']):
				testX.append(data)
				testY.append(target)

		print(f"{len(testX)} features to test from {len(test)} patients")

		#get predictions for each instance in validation data
		pred = model.predict(testX)
		print(metrics.classification_report(testY, pred, zero_division=0))
		break

def main():
	runSVM(getTopHat())
	runSVM(getLBP())


if __name__ == "__main__":
	main()
