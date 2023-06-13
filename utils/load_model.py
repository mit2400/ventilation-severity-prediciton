import tensorflow as tf
import numpy as np
import pickle
from sklearn.utils import class_weight

from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import metrics, regularizers


def get_uncompiled_model(input_shape=(6,30), drop_rate=0.4, num_units=16, num_layer=3, reg = 0.01):
    x_input = Input(shape=(input_shape)) # (6, 32)
    x = Dropout(drop_rate)(x_input)
    for i in range(int(num_layer-1)):
        x = Bidirectional(LSTM(units=num_units, return_sequences=True))(x)
        x = Dropout(drop_rate)(x)
    x = Bidirectional(LSTM(units=num_units, return_sequences=False))(x)
    x = Dropout(drop_rate)(x)
    ouput = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(reg))(x)
    model = Model(inputs=x_input, outputs=ouput, name='severity_prediction')
    return model

def get_compiled_model(input_shape=(6,30), configs=None):
    if configs==None:
        configs = {
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
    model = get_uncompiled_model(input_shape, configs['drop_rate'], configs['num_units'], configs['num_layer'], configs['regularize'])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=configs['learning_rate']), 
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=configs['label_smoothing']),
        metrics=[metrics.AUC(name='auc', curve='ROC'), metrics.AUC(name='ap', curve='PR')]   
    )
    return model