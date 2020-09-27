import os
import sys
import numpy as np
from numpy import interp
from itertools import cycle
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, auc, roc_curve

def binariza(label,n_classes):
    label = int(label)
    label_binary = np.zeros(n_classes,dtype='int8')
    label_binary[label] = 1
    return label_binary

predictions = {}

for root,dir,files in os.walk('./predictions'):
    for file in files:
        predictions[file.replace('.npz','').replace('svm_','')] = np.load(root+'/'+file,allow_pickle=True)

for prediction in predictions.keys():
    print(prediction)
    preds = predictions[prediction]['preds']
    values = predictions[prediction]['values']
    scores = predictions[prediction]['scores']
    
    bpreds = []
    bvalues = []
    bscores = []
    
    n_classes = len(scores[0][0])

    for value,pred,score in zip(values,preds,scores):
        bvalue = np.ndarray((len(value),n_classes),dtype='int8')
        bpred = np.ndarray((len(pred),n_classes),dtype='int8')
        for i,v,p,s in zip(range(len(value)),value,pred,score):
            bvalue[i] = binariza(v,n_classes)
            bpred[i] = binariza(p,n_classes)
        
            bpreds.append(bpred[i])
            bvalues.append(bvalue[i])
            bscores.append(s)

    
    bvalues = np.array(bvalues)
    bpreds = np.array(bpreds)
    bscores = np.array(bscores,'object')

    # print(metrics.classification_report(bvalues,bpreds,zero_division=False))
    # print('accuracy_score: ', metrics.accuracy_score(bvalues,bpreds))
    
    sum_scores = np.sum(bscores,axis=0)
    mean_scores = sum_scores / len(bvalues)
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(bvalues[:, i], bscores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(bvalues.ravel(), bscores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    plt.figure()
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))
        
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC multi-class for predictions of extractor {prediction}')
    plt.legend(loc="lower right")
    plt.show()


