import tensorflow as tf
import numpy as np
import pickle
from sklearn.utils import class_weight

def get_datasets(test = False, batch_size = 128):
    if test:
        with open("data/X_sample.pkl", "rb") as f:            X = pickle.load(f)
        with open("data/Y_sample.pkl", "rb") as f:            Y = pickle.load(f)
    else:
        with open("data/X.pkl", "rb") as f:            X = pickle.load(f)
        with open("data/Y.pkl", "rb") as f:            Y = pickle.load(f)
    print("Train shape: {}, {}, Valid shape: {}, {}".format(X[0].shape, Y[0].shape, X[1].shape, Y[1].shape))
    class_weights = []
    for i in range(4):
        class_weights.append(class_weight.compute_class_weight(class_weight ='balanced', classes=np.unique(Y[i]), y =Y[i]))
    print(f"Classs weights\t", end='')
    print(class_weights)

    datasets = [[] for _ in range(4)]
    for i in range(4):
        datasets[i]= tf.data.Dataset.from_tensor_slices((X[i], Y[i]))
        if i == 0:
            datasets[i]= datasets[i].shuffle(buffer_size=8092).batch(batch_size)
        else:
            datasets[i]= datasets[i].batch(batch_size) 
    
    return datasets, class_weights