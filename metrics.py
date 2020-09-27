import os
import sys
import seaborn as sn
import numpy as np
import pandas as pd
from numpy import interp
from itertools import cycle
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_curve

def binariza(label,n_classes):
    label = int(label)
    label_binary = np.zeros(n_classes,dtype='int8')
    label_binary[label] = 1
    return label_binary

if not os.path.isdir('results'):
    os.mkdir('results')

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
    
    cm = np.ndarray((n_classes,n_classes),dtype='uint16')

    for value,pred,score in zip(values,preds,scores):
        bvalue = np.ndarray((len(value),n_classes),dtype='int8')
        bpred = np.ndarray((len(pred),n_classes),dtype='int8')
        for i,v,p,s in zip(range(len(value)),value,pred,score):
            bvalue[i] = binariza(v,n_classes)
            bpred[i] = binariza(p,n_classes)

            cm[int(p)][int(v)] += 1

            bpreds.append(bpred[i])
            bvalues.append(bvalue[i])
            bscores.append(s)


    bvalues = np.array(bvalues)
    bpreds = np.array(bpreds)
    bscores = np.array(bscores,'object')

    with open(f'results/{prediction}_report.txt','w') as out:
        sum_scores = np.sum(bscores,axis=0)
        mean_scores = sum_scores / len(bvalues)
        out.write('{} report\n'.format(prediction))
        out.write(metrics.classification_report(bvalues,bpreds,zero_division=False))
        out.write('\n')
        out.write('accuracy_score: {}'.format(metrics.accuracy_score(bvalues,bpreds)))

    #create confusion_matrix
    row_sums = cm.sum(axis=1)
    column_sums = cm.sum(axis=0)

    labels = [str(i) for i in range(n_classes)]
    sensibilidades = []
    especificidades = []
    for label in range(6):
        tp = cm[label][label]
        fp = row_sums[label] - tp

        #sensibilidade = tp/(tp+fp)
        soma = np.sum(cm[label])
        if soma != 0:
            sensibilidade = tp / soma
        else:
            sensibilidade = 0
        sensibilidades.append(sensibilidade)

        #especificidade = tn/(tn+fn)
        #tn = 0
        tn = np.sum(np.delete(np.delete(cm, label, 0), label, 1))
        fn = column_sums[label] - tp

        especificidade = tn / (tn + fn)
        especificidades.append(especificidade)

    df = pd.DataFrame.from_dict({"sensibilidades":sensibilidades, "especificidades": especificidades})
    df.to_csv(f"results/{prediction}.csv",index=labels,index_label='labels')

    df_cm = pd.DataFrame(cm, range(6), range(6))
    plt.figure(figsize=(6,6))
    ax = plt.axes()
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, fmt='g', ax=ax) # font size
    plt.title(f'Confusion Matrix {prediction}')
    plt.ylabel('True Label')
    plt.xlabel('Predicated Label')
    plt.savefig(f"results/{prediction}_cm.png")
    plt.clf()


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
    
    plt.figure(figsize=(8,8))
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
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC multi-class {prediction}')
    plt.legend(loc="lower right")
    plt.savefig(f"results/{prediction}_roc.png")
    plt.clf()


