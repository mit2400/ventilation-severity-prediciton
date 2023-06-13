import os
import json
import tensorflow as tf
from utils.utils import get_args, get_configs, set_seed, save_hist, plot_history, save_config
from utils.load_data import get_datasets
from utils.load_model import get_uncompiled_model, get_compiled_model
from utils.custum_callbacks import MultiValidationCallback

def main():
    set_seed()
    
    # load configs
    args = get_args()
    configs, summary_filepath = get_configs(args)

    # load data
    datasets, class_weights = get_datasets(test=args.testcode, batch_size=configs['batch_size'])
    input_shape = datasets[0].element_spec[0].shape[-2:] #(6,30)

    #load model
    model = get_compiled_model(input_shape, configs)
    model.summary()
    
    #set callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(summary_filepath,"checkpoint/run_{epoch}"),
            save_best_only=True,save_weights_only=True,monitor="val_auc",mode='max',verbose=0, save_freq='epoch', options=None
        ),
        tf.keras.callbacks.TensorBoard(log_dir=summary_filepath),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.3, min_lr=1e-4, patience=5, verbose=True),
        MultiValidationCallback([datasets[2], datasets[3]])
    ]

    # model fit
    hist=model.fit(datasets[0], validation_data=datasets[1], class_weight=dict(enumerate(class_weights[configs['class_weight']])), 
            epochs=configs['epochs'], callbacks=callbacks, verbose=0)

    #model fit
    plot_history(hist,summary_filepath)
    save_hist(hist,summary_filepath)
    save_config(configs,summary_filepath)

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # do not show tensorflow INFO and WARNING log
    print("TensorFlow version:", tf.__version__)
    main()
