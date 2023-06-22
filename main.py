import os
import json
import tensorflow as tf
from utils.utils import get_args, get_configs, set_seed, save_hist, plot_history, save_config
from utils.load_data import get_datasets
from utils.load_model import get_uncompiled_model, get_compiled_model, get_trained_model_h5, get_trained_model_ckpt
from utils.custum_callbacks import MultiValidationCallback
from tensorflow.keras import metrics

def train(args):
    # load configs
    configs, summary_filepath = get_configs(args)
    print(args)
    print(configs)
    print(summary_filepath)

    # load data
    datasets, class_weights = get_datasets(configs)
    configs['input_shape'] = tuple(datasets[0].element_spec[0].shape[-2:].as_list()) #(6,30)

    #load model
    model = get_compiled_model(configs)
    
    #set callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(summary_filepath,"checkpoint/run_{epoch}"),
            save_best_only=False,save_weights_only=True,monitor="val_auc",
            mode='max',verbose=0, save_freq='epoch', options=None
        ),
        tf.keras.callbacks.TensorBoard(log_dir=summary_filepath),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', mode='max', 
                        factor=0.3, min_lr=1e-6, patience=3, verbose=True)
        # MultiValidationCallback([datasets[2], datasets[3]])
    ]

    # model fit
    hist=model.fit(datasets[0], validation_data=datasets[1], 
            class_weight=dict(enumerate(class_weights[configs['class_weight']])), 
            epochs=configs['epochs'], callbacks=callbacks, verbose=1)

    #save model history
    print(f'Saving model to {summary_filepath}')
    plot_history(hist,summary_filepath)
    save_hist(hist,summary_filepath)
    save_config(configs,summary_filepath)

def eval(args):
    if args.eval_path.endswith("h5"):
        model = get_trained_model_h5(args.eval_path)    
    else:
        model = get_trained_model_ckpt(args.eval_path,args.eval_epoch)
    model.summary()

    drop_scores=False
    if model.input_shape[2] == 25:
        drop_scores=True

    datasets, class_weights = get_datasets(eval=True, drop_scores=True)
    for i in range(len(datasets)):
        model.evaluate(datasets[i])
    #todo

def tuning(args):
    # if args.eval_path.endswith("h5"):
    #     model = get_trained_model_h5(args.eval_path)    
    # else:
    #     model = get_trained_model_ckpt(args.eval_path,args.eval_epoch)
    # model.summary()

    configs = {
            "input_shape": (6,25),
            "drop_rate": 0.4,
            "num_units": 8,
            "num_layer": 2,
            "regularize": 0.01,
            "learning_rate": 0.001,
            "label_smoothing": 0.2,
            "class_weight": 0,
            "batch_size": 128,
            "epochs": 20
    }

    model = get_compiled_model(configs=configs)

    drop_scores=False
    if model.input_shape[2] == 25:
        drop_scores=True

    datasets, class_weights = get_datasets(drop_scores=True, tuning=True)
    #datasets[3] ICU_train, datasets[4] ICU_valid

    # # Set all layers to be non-trainable (weights will not be updated)
    # for layer in model.layers:
    #     layer.trainable = False
    # # Specify the layers that you want to train (replace 'layer_to_train' with the name or index of the layer)
    # model.get_layer('dense_5').trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.2),
        metrics=[metrics.AUC(name='auc', curve='ROC'), metrics.AUC(name='ap', curve='PR')]
    )
    
    summary_filepath=f"./logs/tuning_0.2_from_bottom"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(summary_filepath,"checkpoint/run_{epoch}"),
            save_best_only=True,save_weights_only=False,monitor="val_auc",mode='max',verbose=0, save_freq='epoch', options=None
        ),
        tf.keras.callbacks.TensorBoard(log_dir=summary_filepath),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.3, min_lr=1e-6, patience=10, verbose=True),
        MultiValidationCallback([datasets[1], datasets[3]])
    ]

    hist=model.fit(datasets[4], validation_data=datasets[5], class_weight=dict(enumerate(class_weights[3])), 
            epochs=200, callbacks=callbacks, verbose=0)
    print(f'Saving model to {summary_filepath}')
    plot_history(hist,summary_filepath)
    save_hist(hist,summary_filepath)

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
    print("TensorFlow version:", tf.__version__)
    set_seed()
    args = get_args()
    if args.eval:       eval(args)
    elif args.tuning:   tuning(args)
    else:               train(args)
