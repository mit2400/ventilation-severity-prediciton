import tensorflow as tf
import numpy as np
import pickle
from sklearn.utils import class_weight

def get_datasets(configs=None,eval=False,drop_scores=False):
    if configs == None: #eval cases
        configs = {}
        configs['isTest'] = False
        configs['batch_size']=128
        configs['impute']=0

    if configs['isTest']:
        X_npz = np.load('data/X_sample.npz')
        Y_npz = np.load('data/Y_sample.npz')
    else:
        #meaning three type of time-series dataprocessing method
        # 0: fill empty value with reference value
        # 1: backfill values for 6 hours and fill ref. val.
        # 2: backfill values for 6 hours and drop rows with only small value available and then fill ref. val.
        if configs['impute'] == 0:
            X_npz = np.load('data/X.npz')
            Y_npz = np.load('data/Y.npz')
        elif configs['impute'] == 1:
            X_npz = np.load('data/X_b6.npz')
            Y_npz = np.load('data/Y_b6.npz')
        elif configs['impute'] == 2:
            X_npz = np.load('data/X_b6_drop.npz')
            Y_npz = np.load('data/Y_b6_drop.npz')
        else:
            print("fail to load")
    X = [X_npz['X_{}'.format(i)] for i in range(len(X_npz))]
    Y = [Y_npz['Y_{}'.format(i)] for i in range(len(Y_npz))]
    
    if drop_scores:
        for i in range(4):
            X[i] = X[i][:, :, :-5]

    class_weights = []
    for i in range(4):
        class_weights.append(class_weight.compute_class_weight(class_weight ='balanced', classes=np.unique(Y[i]), y =Y[i]))
    
    datasets = [[] for _ in range(4)]
    for i in range(4):
        datasets[i]= tf.data.Dataset.from_tensor_slices((X[i], Y[i]))
        if i == 0:
            datasets[i]= datasets[i].shuffle(buffer_size=8092).batch(configs['batch_size'])
        else:
            datasets[i]= datasets[i].batch(configs['batch_size']) 
    
    print('load dataset as tensorflow Dataset object')
    print("Train shape: {}, {}, Valid shape: {}, {}".format(X[0].shape, Y[0].shape, X[1].shape, Y[1].shape))
    print("Classs weights ", class_weights)
    
    if eval:  
        return datasets[1:], class_weights[1:]
    return datasets, class_weights