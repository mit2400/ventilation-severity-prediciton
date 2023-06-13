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

def get_compiled_model(input_shape=(6,30), drop_rate=0.4, num_units=16,num_layer=3, reg = 0.01, learning_rate=1e-3,label_smoothing=0.2):
    model = get_uncompiled_model(input_shape, drop_rate, num_units, num_layer, reg)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),
        metrics=[metrics.AUC(name='auc', curve='ROC'), metrics.AUC(name='ap', curve='PR')]   
    )
    return model