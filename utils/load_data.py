import tensorflow as tf
import numpy as np
import pickle
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
import os 
from pathlib import Path

def get_datasets(configs=None,eval=False,drop_scores=False, tuning=False):
    abs_path = Path(__file__).parent.parent
    if configs == None: #eval cases
        configs = {}
        configs['isTest'] = False
        configs['batch_size']=128
        configs['impute']=0

    if configs['isTest']:
        X_npz = np.load(os.path.join(abs_path, 'data/X_sample.npz'))
        Y_npz = np.load(os.path.join(abs_path, 'data/Y_sample.npz'))
    else:
        # Meaning three type of time-series data processing method
        # 0: fill empty value with reference value
        # 1: backfill values for 6 hours and fill ref. val.
        # 2: backfill values for 6 hours and drop rows with only small value available and then fill ref. val.
        if configs['impute'] == 0:
            X_npz = np.load(os.path.join(abs_path, 'data/X.npz'))
            Y_npz = np.load(os.path.join(abs_path, 'data/Y.npz'))
        elif configs['impute'] == 1:
            X_npz = np.load(os.path.join(abs_path, 'data/X_b6.npz'))
            Y_npz = np.load(os.path.join(abs_path, 'data/Y_b6.npz'))
        elif configs['impute'] == 2:
            X_npz = np.load(os.path.join(abs_path, 'data/X_b6_drop.npz'))
            Y_npz = np.load(os.path.join(abs_path, 'data/Y_b6_drop.npz'))
        else:
            print("fail to load")
    X = [X_npz['X_{}'.format(i)] for i in range(len(X_npz))]
    Y = [Y_npz['Y_{}'.format(i)] for i in range(len(Y_npz))]
    
    if drop_scores:
        for i in range(len(X)):
            X[i] = X[i][:, :, :-5]

    if tuning:
        train_data, test_data, train_labels, test_labels = train_test_split(X[2], Y[2], test_size=0.9)
        X.append(train_data)
        X.append(test_data)
        Y.append(train_labels)
        Y.append(test_labels)

    class_weights = []
    for i in range(len(X)):
        class_weights.append(
            class_weight.compute_class_weight(class_weight ='balanced', classes=np.unique(Y[i]), y =Y[i]))
    

    datasets = [[] for _ in range(len(X))]
    for i in range(len(X)):
        datasets[i]= tf.data.Dataset.from_tensor_slices((X[i], Y[i]))
        if i == 0:
            datasets[i]= datasets[i].shuffle(buffer_size=8092).batch(configs['batch_size']).prefetch(1)
        else:
            datasets[i]= datasets[i].batch(configs['batch_size']) 
    
    print('load dataset as tensorflow Dataset object')
    print("Train shape: {}, Valid shape: {}, ICU shape {}, COVID shape{}, Test shape{}".format(X[0].shape, X[1].shape,X[2].shape,X[3].shape,X[4].shape))
    print("Classs weights ", class_weights)
    
    if eval:  
        return datasets[1:], class_weights[1:]
    return datasets, class_weights

def get_datasets_np(configs=None,eval=False,drop_scores=False, tuning=False):
    abs_path = Path(__file__).parent.parent
    
    if configs == None: #eval cases
        configs = {}
        configs['isTest'] = False
        configs['batch_size']=128
        configs['impute']=0

    if configs['isTest']:
        X_npz = np.load(os.path.join(abs_path, 'data/X_sample.npz'))
        Y_npz = np.load(os.path.join(abs_path, 'data/Y_sample.npz'))
    else:
        # Meaning three type of time-series data processing method
        # 0: fill empty value with reference value
        # 1: backfill values for 6 hours and fill ref. val.
        # 2: backfill values for 6 hours and drop rows with only small value available 
        # and then fill ref. val.
        if configs['impute'] == 0:
            X_npz = np.load(os.path.join(abs_path, 'data/X.npz'))
            Y_npz = np.load(os.path.join(abs_path, 'data/Y.npz'))
        elif configs['impute'] == 1:
            X_npz = np.load(os.path.join(abs_path, 'data/X_b6.npz'))
            Y_npz = np.load(os.path.join(abs_path, 'data/Y_b6.npz'))
        elif configs['impute'] == 2:
            X_npz = np.load(os.path.join(abs_path, 'data/X_b6_drop.npz'))
            Y_npz = np.load(os.path.join(abs_path, 'data/Y_b6_drop.npz'))
        else:
            print("fail to load")
    X = [X_npz['X_{}'.format(i)] for i in range(len(X_npz))]
    Y = [Y_npz['Y_{}'.format(i)] for i in range(len(Y_npz))]
    
    if drop_scores:
        for i in range(len(X)):
            X[i] = X[i][:, :, :-5]

    if tuning:
        train_data, test_data, train_labels, test_labels = train_test_split(X[2], Y[2], test_size=0.9)
        X.append(train_data)
        X.append(test_data)
        Y.append(train_labels)
        Y.append(test_labels)
        
    class_weights = []
    for i in range(len(X)):
        class_weights.append(class_weight.compute_class_weight(class_weight ='balanced', 
                                                               classes=np.unique(Y[i]), y =Y[i]))
       
    if eval:  
        return X[1:],Y[1:], class_weights[1:]
    return X, Y, class_weights