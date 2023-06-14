import tensorflow as tf
import numpy as np
import pickle
from sklearn.utils import class_weight

def get_datasets(batch_size = 128,test = False,eval=False,drop_scores=False):
    if test:
        X_npz = np.load('data/X_sample.npz')
        Y_npz = np.load('data/Y_sample.npz')
        X = [X_npz['X_{}'.format(i)] for i in range(len(X_npz))]
        Y = [Y_npz['Y_{}'.format(i)] for i in range(len(Y_npz))]
    else:
        X_npz = np.load('data/X.npz')
        Y_npz = np.load('data/Y.npz')
        X = [X_npz['X_{}'.format(i)] for i in range(len(X_npz))]
        Y = [Y_npz['Y_{}'.format(i)] for i in range(len(Y_npz))]
    
    if drop_scores==True:
        for i in range(4):
            X[i] = X[i][:, :, :-5]

    class_weights = []
    for i in range(4):
        class_weights.append(class_weight.compute_class_weight(class_weight ='balanced', classes=np.unique(Y[i]), y =Y[i]))
    
    datasets = [[] for _ in range(4)]
    for i in range(4):
        datasets[i]= tf.data.Dataset.from_tensor_slices((X[i], Y[i]))
        if i == 0:
            datasets[i]= datasets[i].shuffle(buffer_size=8092).batch(batch_size)
        else:
            datasets[i]= datasets[i].batch(batch_size) 
    
    print('load dataset as tensorflow Dataset object')
    print("Train shape: {}, {}, Valid shape: {}, {}".format(X[0].shape, Y[0].shape, X[1].shape, Y[1].shape))
    print("Classs weights ", class_weights)
    
    if eval == True:
        return datasets[1:], class_weights[1:]
    return datasets, class_weights