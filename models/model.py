import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # do not show tensorflow INFO log

import sys
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.layers import LSTM, Bidirectional, Concatenate, GRU
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization, GlobalMaxPooling1D
from tensorflow.keras.layers import Dense, Flatten, Conv2D


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import metrics, regularizers

class SeverityModel:
    def __init__(self, config):
        self.config = config
        # self.build_model()
        # self.init_saver()

    def build_model(self):

        #todo, set those as config
        device = 'GPU:0'
        dp = 0.4
        nunit = 16
        lr=1e-3
        ls=0.2
        epoch=10
        input_shape = (6,30)
        
        with tf.device(device):
            # x_input = Input(shape=(X[0].shape[-2:])) # (6, 25)
            x_input = Input(shape=input_shape) # todo
            d0 = Dropout(dp)(x_input)
            x1 = Bidirectional(LSTM(units=nunit, return_sequences=True))(d0)
            d1 = Dropout(dp)(x1)
            l1 = Bidirectional(LSTM(units=nunit, return_sequences=True))(d1)
            d2 = Dropout(dp)(l1)
            l2 = Bidirectional(LSTM(units=nunit, return_sequences=False))(d2)
            d3 = Dropout(dp)(l2)
            d4 = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))(d3)
            model = Model(inputs=x_input, outputs=d4, name='Bidirectional_LSTM')
        return model
    
class MVModel(Model):
    def __init__(self,config = []):
        super(MVModel, self).__init__()
        self.config = config

        self.drop1 = Dropout(config['drop_rate'])
        self.drop2 = Dropout(config['drop_rate'])
        self.drop3 = Dropout(config['drop_rate'])
        self.drop4 = Dropout(config['drop_rate'])
        self.BiLSTM1 = Bidirectional(LSTM(units=config['n_units'], return_sequences=True))
        self.BiLSTM2 = Bidirectional(LSTM(units=config['n_units'], return_sequences=True))
        self.BiLSTM_out = Bidirectional(LSTM(units=config['n_units'], return_sequences=False))
        self.dense = Dense(1, activation=config['activation'], kernel_regularizer=regularizers.l2(config['regularizer']))

    def call(self, x):
        x=self.drop1(x)
        x=self.BiLSTM1(x)
        x=self.drop2(x)
        x=self.BiLSTM2(x)
        x=self.drop3(x)
        x=self.BiLSTM_out(x)
        x=self.drop4(x)
        x=self.dense(x)
        return x


    def init_saver(self):
        pass

if __name__ == '__main__':

    tf.debugging.set_log_device_placement(True)

    config = {}
    config['device'] = 'GPU:0'
    config['n_units'] = 16 # in lstm layer
    config['drop_rate'] = 0.4
    config['learning_rate'] = 16
    config['label_smoothing'] = 1e-3
    config['activation'] = 'sigmoid'
    config['regularizer'] = 0.01

    test = SeverityModel(config)
    model = test.build_model()
    model.summary()
    print(model)

    with tf.device('GPU:0'):
        model = MVModel(config)
    print(model)






