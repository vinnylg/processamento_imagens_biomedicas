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
from sklearn.metrics import confusion_matrix
from extraction import getLBP, getTopHat, getFractalDim

def print_confusion_matrix (cm):
    for row in cm:
        print('\t',row)

def classify(dataset):
    if(not os.path.isdir('./predictions')):
        os.mkdir('./predictions')

    scores = []
	#prepare model
    model = svm.SVC(kernel='rbf', random_state=0, gamma='scale',C=1, decision_function_shape='ovo', probability=True)

    cm = [[0 for i in range(6)] for j in range(6)]
    #print_confusion_matrix(cm)
	#for each patient
    for index in range(len(dataset)):
        start_time = time.time()
        preds = []
        values = []

		#split dataset into validation set and training set
		#leave one out
        train = dataset[:index] + dataset[index+1:]
        test = [dataset[index]]
		#k-fold simples
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
        if trainX.ndim == 1:
            trainX = trainX.reshape(-1,1)
        else:
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
        if testX.ndim == 1:
            testX = testX.reshape(-1,1)
        else:
            testX = normalize(testX)

        testY = np.array(testY, dtype='float')

        print(f"{testX.shape} features to test for patient {index}")

		#get predictions for each instance in validation data
        pred = model.predict(testX)
        prob = model.predict_proba(testX)
        #print('prob:', prob)
        for label, predicted in zip(testY, pred):
            #print('label:', label)
            #print('predicted:', predicted)
            cm[int(predicted)][int(label)] += 1

        #print('cm')
        #print_confusion_matrix(cm)

        preds.append(prob)
        values.append(testY)
        scores.append(metrics.accuracy_score(testY, pred))
        print(f"accuracy: {metrics.accuracy_score(testY, pred)}")
        print(f" Time elapsed: {datetime.timedelta(seconds=round(time.time()-start_time))} seconds")

    #print(np.mean(scores))
    #print_confusion_matrix(cm)

    return preds, values, cm, np.mean(scores)

def main():
    preds, values, cm, acc = classify(getTopHat())
    with open('./predictions/svm_tophat.npz', 'wb') as out:
       np.savez(out, preds=preds,values=values, cm=cm, acc=acc)
       print(f"./predictions/svm_tophat.npz saved")

    # preds, values, cm, acc = classify(getFractalDim())
    # with open('./predictions/svm_fractaldim.npz', 'wb') as out:
    #     np.savez(out, preds=preds,values=values, cm=cm, acc=acc)
    #     print(f"./predictions/svm_fractaldim.npz saved")

	# preds, values = runSVM(getLBP())
	# with open('./predictions/svm_LBP.npz', 'wb') as out:
	# 	np.savez(out, preds=preds,values=values, cm=cm, acc=acc)
	# 	print(f"./predictions/svm_LBP.npz saved")

if __name__ == "__main__":
	main()
