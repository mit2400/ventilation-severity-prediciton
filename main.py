import os
import tensorflow as tf
from utils.utils import set_seed, save_hist, plot_history
from utils.load_data import get_datasets
from utils.load_model import get_uncompiled_model, get_compiled_model
from utils.custum_callbacks import MultiValidationCallback

def main():
    set_seed()

    # load data
    datasets, class_weights = get_datasets(test=True)
    input_shape = datasets[0].element_spec[0].shape[-2:] #(6,30)

    #### todo, parameters to search
    dp = 0.4
    nunit = 8
    nlayer = 2
    reg = 0.01
    lr = 1e-3
    ls = 0.2
    cw = 2
    epochs=20
    
    #load model
    model = get_compiled_model(input_shape, dp,nunit,nlayer,reg,lr,ls)
    model.summary()
    
    #callbacks
    summary_filepath = f"./logs/model_u{nunit}_l{nlayer}_d{dp}_r{reg}_lr{lr}_ls{ls}_cw{cw}"
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(summary_filepath,"checkpoint/run_{epoch}"),
            save_best_only=True,save_weights_only=True,monitor="val_auc",mode='max',verbose=0, save_freq='epoch', options=None
        ),
        tf.keras.callbacks.TensorBoard(log_dir=summary_filepath),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.3, min_lr=1e-4, patience=5, verbose=True),
        MultiValidationCallback([datasets[2], datasets[3]])
    ]

    hist=model.fit(datasets[0], validation_data=datasets[1], class_weight=dict(enumerate(class_weights[cw])), 
            epochs=epochs, callbacks=callbacks, verbose=0)

    plot_history(hist,summary_filepath)
    save_hist(hist,summary_filepath)

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # do not show tensorflow INFO and WARNING log
    print("TensorFlow version:", tf.__version__)
    main()
