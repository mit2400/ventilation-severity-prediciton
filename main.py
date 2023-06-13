
# from utils.config import process_config
# from utils.dirs import create_dirs
# from utils.logger import Logger
# from utils.utils import get_args

import os
import warnings
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # do not show tensorflow INFO and WARNING log
warnings.filterwarnings('ignore') # do not show tensorflow INFO and WARNING log
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import time
import tensorflow as tf
import random
import numpy as np
import pickle

import matplotlib.pyplot as plt
from models.model import MVModel
from eval.evaluation import eval_severity_scores

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Concatenate, GRU
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import metrics, regularizers
from sklearn.utils import class_weight
from tqdm import tqdm

def set_seed(seed=42):
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    # scikit learn use numpy random generator

def main():
    set_seed()
    
    config = {}
    config['device'] = 'GPU:0'
    config['n_units'] = 16 # in lstm layer
    config['drop_rate'] = 0.4
    config['learning_rate'] = 16
    config['label_smoothing'] = 1e-3
    config['activation'] = 'sigmoid'
    config['regularizer'] = 0.01

    # load data
    test = False
    if test:
        with open("data/X_sample.pkl", "rb") as f:            X = pickle.load(f)
        with open("data/Y_sample.pkl", "rb") as f:            Y = pickle.load(f)
    else:
        with open("data/X.pkl", "rb") as f:            X = pickle.load(f)
        with open("data/Y.pkl", "rb") as f:            Y = pickle.load(f)
    print("Train shape: {}, {}, Valid shape: {}, {}".format(X[0].shape, Y[0].shape, X[1].shape, Y[1].shape))
    class_weights = class_weight.compute_class_weight(class_weight ='balanced', classes=np.unique(Y[0]), y =Y[0])
    print(f'class weight: {class_weights}')

    datasets = [[] for _ in range(4)]
    batch_size = 128
    for i in range(4):
        datasets[i]= tf.data.Dataset.from_tensor_slices((X[i], Y[i]))
        if i == 0:
            datasets[i]= datasets[i].shuffle(buffer_size=8092).batch(batch_size)
        else:
            datasets[i]= datasets[i].batch(batch_size) 

    def get_uncompiled_model(drop_rate=0.4, num_units=16, reg = 0.01):
        x_input = Input(shape=(X[0].shape[-2:])) # (6, 32)
        d0 = Dropout(drop_rate)(x_input)
        x1 = Bidirectional(LSTM(units=num_units, return_sequences=True))(d0)
        d1 = Dropout(drop_rate)(x1)
        l1 = Bidirectional(LSTM(units=num_units, return_sequences=True))(d1)
        d2 = Dropout(drop_rate)(l1)
        l2 = Bidirectional(LSTM(units=num_units, return_sequences=False))(d2)
        d3 = Dropout(drop_rate)(l2)
        d4 = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(reg))(d3)
        model = Model(inputs=x_input, outputs=d4, name='Bidirectional_LSTM')
        return model
    
    def get_compiled_model(drop_rate=0.4, num_units=16, reg = 0.01, learning_rate=1e-3,label_smoothing=0.2):
        model = get_uncompiled_model(drop_rate, num_units, reg)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
            loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),
            metrics=[metrics.AUC(name='auc', curve='ROC'), metrics.AUC(name='ap', curve='PR')]   
        )
        return model

    #todo
    dp = 0.4
    nunit = 8
    reg = 0.01
    lr = 1e-3
    ls = 0.2
    epochs=20
    
    model = get_compiled_model(dp,nunit,reg,lr,ls)
    model.summary()
    
    checkpoint_filepath = f"./result/Model_{dp}_{nunit}_{lr}"
    sv = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_filepath,"run_{epoch}"),
        save_best_only=False,save_weights_only=True,monitor="val_auc",mode='max',verbose=0, options=None)
    tb = tf.keras.callbacks.TensorBoard(log_dir=f"./summaries/Model_{dp}_{nunit}_{lr}") 
    lr_sched = ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.3, min_lr=1e-4, patience=5, verbose=True)

    class MultiValidationCallback(tf.keras.callbacks.Callback):
        def __init__(self, val_datasets):
            super().__init__()
            self.val_datasets = val_datasets

        def on_epoch_end(self, epoch, logs=None):
            # lr = self.model.optimizer.lr
            # If the learning rate is a decayed learning rate, we need to compute it
            # if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            #     lr = lr(epoch)
            # print(f'learning rate: {lr}')
            metrics = [[] for _ in range(3)]
            for i, val_dataset in enumerate(self.val_datasets):
                val_loss, val_auc, val_ap = self.model.evaluate(val_dataset,verbose=0)
                logs[f'val{i+2}_loss'] = val_loss
                logs[f'val{i+2}_auc'] = val_auc
                logs[f'val{i+2}_ap'] = val_ap
            formatted_items = [f'{key}: {value:.3f}' for key, value in logs.items()]
            print(formatted_items)
    mv = MultiValidationCallback([datasets[2], datasets[3]])

    hist=model.fit(datasets[0], validation_data=datasets[1], class_weight=dict(enumerate(class_weights)), 
            epochs=epochs, callbacks=[sv,mv,tb,lr_sched], verbose=0)
    print(hist.params)

    def plot_metrics(history):
        metrics = ['loss', 'auc', 'ap']
        plt.figure(figsize=(15, 10))
        for n, metric in enumerate(metrics):
            name = metric.replace("_"," ").capitalize()
            plt.subplot(1,3,n+1)
            plt.plot(history.epoch, history.history[metric], label='Train')
            plt.plot(history.epoch, history.history['val_'+metric], linestyle="--", label='Val')
            plt.plot(history.epoch, history.history['val2_'+metric], linestyle="--", label='Val2')
            plt.plot(history.epoch, history.history['val3_'+metric], linestyle="--", label='Val3')
            plt.xlabel('Epoch')
            plt.ylabel(name)
            plt.legend()
            checkpoint_filepath = f"./summaries/Model_{dp}_{nunit}_{lr}"
            plt.savefig(os.path.join(checkpoint_filepath,"fig.pdf"),bbox_inches='tight')

    plot_metrics(hist)

    # 12h 24h
    #0.819 0.808
    #0.792 0.789
    #0.835 0.834

    y_valid=[]
    y_prob=[]
    for i in range(1,4):
        y_prob.append(model.predict(X[i]).squeeze())  
        y_valid.append(Y[i])

if __name__ == '__main__':
    print("TensorFlow version:", tf.__version__)
    main()
