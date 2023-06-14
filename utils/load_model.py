import tensorflow as tf
import numpy as np
import pickle
from sklearn.utils import class_weight

from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import metrics, regularizers

from tensorflow.keras.models import load_model
import os
import json

# def get_uncompiled_model(input_shape=(6,30), drop_rate=0.4, num_units=16, num_layer=3, reg = 0.01):
def get_uncompiled_model(configs):
    if configs==None:
        print("No configuration given")
    x_input = Input(shape=(configs['input_shape'])) # (6, 32)
    x = Dropout(configs['drop_rate'])(x_input)
    for i in range(int(configs['num_layer']-1)):
        x = Bidirectional(LSTM(units=configs['num_units'], return_sequences=True))(x)
        x = Dropout(configs['drop_rate'])(x)
    x = Bidirectional(LSTM(units=configs['num_units'], return_sequences=False))(x)
    x = Dropout(configs['drop_rate'])(x)
    ouput = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(configs['regularize']))(x)
    model = Model(inputs=x_input, outputs=ouput, name='severity_prediction')
    return model

def get_compiled_model(configs):
    if configs==None:
        configs = {
            "input_shape": (6,30),
            "drop_rate": 0.4,
            "num_units": 8,
            "num_layer": 2,
            "regularize": 0.01,
            "learning_rate": 0.001,
            "label_smoothing": 0.2,
            "class_weight": 2,
            "batch_size": 128,
            "epochs": 20
        }
    model = get_uncompiled_model(configs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=configs['learning_rate']),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=configs['label_smoothing']),
        metrics=[metrics.AUC(name='auc', curve='ROC'), metrics.AUC(name='ap', curve='PR')]
    )
    model.summary()
    return model

def get_trained_model_h5(model_path=None):
    if model_path == None:
        model_path = './logs/best_model/lstm_dp04.h5'
    print(f'loading pretrained model from {model_path}')
    return  load_model(model_path)

def get_trained_model_ckpt(model_path=None, epoch=None):
    if model_path == None:
        model_path = './logs/model_drop_rate0.4_num_units8_num_layer2_regularize0.01_learning_rate0.001_label_smoothing0.2_class_weight0_batch_size128_epochs20'
    
    with open(os.path.join(model_path,"config.json"), 'r') as f:
        configs = json.load(f)
    
    model = get_uncompiled_model(configs)
    if epoch == None: # load latest if epoch is not given
        model.load_weights(tf.train.latest_checkpoint(model_path))
        print(f'loading pretrained model from {model_path}')
    else:
        model.load_weights(os.path.join(model_path,f"run_{epoch}"))
        print(f'loading pretrained model from {model_path} at epoch{epoch}')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=configs['learning_rate']),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=configs['label_smoothing']),
        metrics=[metrics.AUC(name='auc', curve='ROC'), metrics.AUC(name='ap', curve='PR')]
    )

    model.summary()
    return  load_model(model_path)