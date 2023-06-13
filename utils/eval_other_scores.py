from sklearn.metrics import roc_auc_score, average_precision_score,accuracy_score
import pandas as pd
import numpy as np

def extract_max(ls):
        max_val = max(max(ls[1]), max(ls[2]), max(ls[0]))
        #print(max_val)
        return max_val

def sigmoid(lst):
    return 1 / (1 + np.exp(-np.array(lst)))

def eval_severity_scores(X,Y):
    columns = ['Rox', 'MEWS', 'NEWS', 'Sofa', 'QSofa']
    rows = ['mimic-train', 'mimic-valid', 'KU-ICU','KU-COVID']

    total_auc = [[] for _ in range(4)]
    total_ap = [[] for _ in range(4)]
    total_acc = [[] for _ in range(4)]

    for i in range(4):
        for j in range(5):
            if j == 0 : # rox index
                y_prob = 1-sigmoid(X[i][:,-1,-5+j])
            else:
                y_prob = sigmoid(X[i][:,-1,-5+j])
            
            y_pred = np.where(np.array(y_prob) >= 0.5, 1, 0)

            total_auc[i].append(roc_auc_score(Y[i], y_prob, average=None))
            total_ap[i].append(average_precision_score(Y[i], y_prob))
            total_acc[i].append(accuracy_score(Y[i], y_pred))
            

    auc = pd.DataFrame(total_auc)
    ap = pd.DataFrame(total_ap)
    acc = pd.DataFrame(total_acc)
    auc.columns = columns
    auc.index = rows
    ap.columns = columns
    ap.index = rows
    acc.columns = columns
    acc.index = rows

    auc.to_csv('logs/severity_auroc.csv')
    ap.to_csv('logs/severity_ap.csv')
    acc.to_csv('logs/severity_acc.csv')

    print('AUROC')
    print(auc)
    print('Average Preicision')
    print(ap)
    



    