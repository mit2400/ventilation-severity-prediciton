
# from utils.config import process_config
# from utils.dirs import create_dirs
# from utils.logger import Logger
# from utils.utils import get_args

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # do not show tensorflow INFO log

import tensorflow as tf
import random
import numpy as np
import pickle

from models.model import MVModel
from eval.evaluation import eval_severity_scores

# from tensorflow.keras.models import Model, Sequential
# from tensorflow.keras.layers import LSTM, Bidirectional, Concatenate, GRU
# from tensorflow.keras.layers import Input, Dense, Dropout
# from tensorflow.keras.layers import BatchNormalization, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import metrics, regularizers
from sklearn.utils import class_weight
from tqdm import tqdm

def set_seed(seed=42):
    random.seed(42)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    # scikit learn use numpy random generator

def main():
    set_seed()
    
    config = []
    # try:
    #     args = get_args()
    #     config = process_config(args.config)
    # except:
    #     print("missing or invalid arguments")
    #     exit(0)
    config = {}
    config['device'] = 'GPU:0'
    config['n_units'] = 16 # in lstm layer
    config['drop_rate'] = 0.4
    config['learning_rate'] = 16
    config['label_smoothing'] = 1e-3
    config['activation'] = 'sigmoid'
    config['regularizer'] = 0.01

    dp = 0.4
    nunit = 16
    lr=1e-3
    ls=0.2
    epoch=20

    # # create the experiments dirs
    # create_dirs([config.summary_dir, config.checkpoint_dir])
    checkpoint_filepath = f"./result/run_{dp}_{nunit}_{lr}.hdf5"
    
    # load data
    with open("data/X.pkl", "rb") as f:
        X = pickle.load(f)
    with open("data/Y.pkl", "rb") as f:
        Y = pickle.load(f)
    
    # with open("data/X_sample.pkl", "rb") as f:
    #     X = pickle.load(f)
    # with open("data/Y_sample.pkl", "rb") as f:
    #     Y = pickle.load(f)

    eval_severity_scores(X,Y)
    class_weights = class_weight.compute_class_weight(class_weight ='balanced', classes=np.unique(Y[0]), y =Y[0])

    model = MVModel(config)
    print(model(X[3][:1]))




    # Instantiate an optimizer.
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
    # Instantiate a loss function.
    loss_fn = tf.keras.losses.BinaryCrossentropy(label_smoothing=ls)

    batch_size = 128
    train_dataset = tf.data.Dataset.from_tensor_slices((X[0], Y[0]))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    
    # Prepare the validation dataset.
    val_dataset = tf.data.Dataset.from_tensor_slices((X[1], Y[1]))
    val_dataset = val_dataset.batch(batch_size)

    epochs = 2
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        # https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch/?hl=ko

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:

                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                logits = model(x_batch_train, training=True)  # Logits for this minibatch

                # Compute the loss value for this minibatch.
                loss_value = loss_fn(y_batch_train, logits)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Log every 200 batches.
            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %s samples" % ((step + 1) * batch_size))





    
    # model.compile(
    #     optimizer=tf.keras.optimizers.Adam(learning_rate=lr), 
    #     loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=ls), 
    #     metrics=[metrics.AUC(name = 'auc')])

    # lr = ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.3, min_lr=1e-6, patience=5, verbose=True)
    # #es = EarlyStopping(monitor='val_auc', mode='max', patience=10, restore_best_weights=True, verbose=True)
    # sv = tf.keras.callbacks.ModelCheckpoint(
    #     checkpoint_filepath, monitor='val_auc', verbose=1, save_best_only=True,
    #     save_weights_only=True, mode='max', save_freq='epoch', options=None
    # )
    
    # print("Train shape: {}, {}, Valid shape: {}, {}".format(X[0].shape, Y[0].shape, X[1].shape, Y[1].shape))
    
    # hist=model.fit(X[0], Y[0], validation_data=(X[1], Y[1]), class_weight=dict(enumerate(class_weights)), 
    #         epochs=epoch, batch_size=128, callbacks=[lr,sv], verbose=True)

    

    # for i in range (1,4):
    #     model.evaluate(X[i], Y[i], verbose=2)

    

    # y_valid=[]
    # y_prob=[]
    # for i in range(1,4):
    #     y_prob.append(model.predict(X[i]).squeeze())  
    #     y_valid.append(Y[i])

    # # create tensorboard logger
    # logger = Logger(sess, config)


if __name__ == '__main__':
    print("TensorFlow version:", tf.__version__)
    main()
