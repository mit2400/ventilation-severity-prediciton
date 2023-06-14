import os
import json
import tensorflow as tf
from utils.utils import get_args, get_configs, set_seed, save_hist, plot_history, save_config
from utils.load_data import get_datasets
from utils.load_model import get_uncompiled_model, get_compiled_model, get_trained_model_h5, get_trained_model_ckpt
from utils.custum_callbacks import MultiValidationCallback

def train(args):
    # load configs
    configs, summary_filepath = get_configs(args)

    # load data
    if args.testcode:
        datasets, class_weights = get_datasets(batch_size = configs['batch_size'],test=True,drop_scores=args.drop_scores)
    else:
        datasets, class_weights = get_datasets(batch_size = configs['batch_size'],drop_scores=args.drop_scores)
    configs['input_shape'] = tuple(datasets[0].element_spec[0].shape[-2:].as_list()) #(6,30)

    #load model
    model = get_compiled_model(configs)
    
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

    drop_scores=False
    if model.input_shape[2] == 25:
        drop_scores=True

    datasets, class_weights = get_datasets(eval=True, drop_scores=drop_scores)
    for i in range(3):
        model.evaluate(datasets[i])
    #todo

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # do not show tensorflow INFO and WARNING log
    print("TensorFlow version:", tf.__version__)
    set_seed()
    args = get_args()
    print(args)
    if args.eval:   eval(args)
    else:           train(args)
