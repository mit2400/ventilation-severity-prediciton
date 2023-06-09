
# from utils.config import process_config
# from utils.dirs import create_dirs
# from utils.logger import Logger
# from utils.utils import get_args

import tensorflow as tf
import numpy as np
import pickle
from models.model import SeverityModel
from eval.evaluation import eval_severity_scores
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Concatenate, GRU
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import metrics, regularizers
from sklearn.utils import class_weight
from tqdm import tqdm

def main():
    
    # capture the config path from the run arguments
    # then process the json configuration file
    config = []
    # try:
    #     args = get_args()
    #     config = process_config(args.config)

    # except:
    #     print("missing or invalid arguments")
    #     exit(0)

    # # create the experiments dirs
    # create_dirs([config.summary_dir, config.checkpoint_dir])

    # # create your data generator
    # data = DataGenerator(config)
    
    # load data
    with open("data/X.pkl", "rb") as f:
        X = pickle.load(f)
    with open("data/Y.pkl", "rb") as f:
        Y = pickle.load(f)
    print(X[0].shape)

    # create an instance of the model you want
    SM = SeverityModel(config)
    model = SM.build_model()    
    
    eval_severity_scores(X,Y)

    y_valid=[]
    y_prob=[]

    class_weights = class_weight.compute_class_weight(class_weight ='balanced', classes=np.unique(Y[0]), y =Y[0])

    dp = 0.4
    nunit = 16
    lr=1e-3
    ls=0.2
    epoch=1
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr), 
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=ls), 
        metrics=[metrics.AUC(name = 'auc')])

    checkpoint_filepath = f"./result/run_{dp}_{nunit}_{lr}.hdf5"
    lr = ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.3, min_lr=1e-6, patience=5, verbose=True)
    #es = EarlyStopping(monitor='val_auc', mode='max', patience=10, restore_best_weights=True, verbose=True)
    sv = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_filepath, monitor='val_auc', verbose=1, save_best_only=True,
        save_weights_only=False, mode='max', save_freq='epoch', options=None
    )
    print("Train shape: {}, {}, Valid shape: {}, {}".format(X[0].shape, Y[0].shape, X[1].shape, Y[1].shape))
    
    hist=model.fit(X[0], Y[0], validation_data=(X[1], Y[1]), class_weight=dict(enumerate(class_weights)), 
            epochs=epoch, batch_size=128, callbacks=[lr,sv], verbose=True)

    for i in range (1,4):
        model.evaluate(X[i], Y[i], verbose=2)

    print(model(X[3][:1]))

    for i in range(1,4):
        y_prob.append(model.predict(X[i]).squeeze())  
        y_valid.append(Y[i])



    # # create tensorboard logger
    # logger = Logger(sess, config)


if __name__ == '__main__':
    print("TensorFlow version:", tf.__version__)
    
    main()
