import os
from sys import exit
import time
import datetime
import numpy as np
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, label_binarize
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import LinearSVC, SVC
import statistics
from extraction import getLBP, getCLBP, getTopHat, getFractalDim

def join_dataset(d1,d2,d3,d4):
    newDataset = []
    for index in range(len(d1)):
        newData = []
        for dd1,dd2,dd3,dd4 in zip(d1[index]['data'],d2[index]['data'],d3[index]['data'],d4[index]['data']):
            dd1.append(dd2)
            dd1.extend(dd3)
            dd1.extend(dd4)
            newData.append(dd1)
        newDataset.append({'data': newData, 'target': d1[index]['target']})
    return newDataset

def classify(dataset):
    absolute_start_time = time.time()
    if(not os.path.isdir('./predictions')):
        os.mkdir('./predictions')
    
    values = []
    preds = []
    scores = []
	#prepare model
    model = LinearSVC(random_state=0, class_weight='balanced', max_iter=3000)
    # model = SVC(kernel='rbf', random_state=0, gamma='scale',C=1, decision_function_shape='ovo')

	#for each patient
    for index in range(len(dataset)):
        start_time = time.time()

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
        score = model.decision_function(testX)
        pred = model.predict(testX)
        
        values.append(testY)
        preds.append(pred)
        scores.append(score)
        
        print(f"accuracy: {accuracy_score(testY, pred)}")
        print(f" Time elapsed: {datetime.timedelta(seconds=round(time.time()-start_time))} seconds")
        print(f" Total time elapsed: {datetime.timedelta(seconds=round(time.time()-absolute_start_time))} seconds")


    return np.array(values,dtype='object'), np.array(preds,dtype='object'), np.array(scores,dtype='object')

def main():
    # values, preds, scores = classify(getTopHat())
    # with open('./predictions/svm_tophat.npz', 'wb') as out:
    #     np.savez(out, values=values, preds=preds, scores=scores)
    #     print(f"./predictions/svm_tophat.npz saved")

    # values, preds, scores = classify(getFractalDim())
    # with open('./predictions/svm_fractaldim.npz', 'wb') as out:
    #     np.savez(out, values=values, preds=preds, scores=scores)
    #     print(f"./predictions/svm_fractaldim.npz saved")

    # values, preds, scores = classify(getLBP())
    # with open('./predictions/svm_LBP.npz', 'wb') as out:
    #     np.savez(out, values=values, preds=preds, scores=scores)
    #     print(f"./predictions/svm_LBP.npz saved")

    # values, preds, scores = classify(getCLBP())
    # with open('./predictions/svm_CLBP.npz', 'wb') as out:
    #     np.savez(out, values=values, preds=preds, scores=scores)
    #     print(f"./predictions/svm_CLBP.npz saved")

    dataset = join_dataset(getTopHat(), getFractalDim(), getLBP(), getCLBP())
    values, preds, scores = classify(dataset)
    with open('./predictions/all.npz', 'wb') as out:
        np.savez(out, values=values, preds=preds, scores=scores)
        print(f"./predictions/all.npz saved")
if __name__ == "__main__":
	main()
