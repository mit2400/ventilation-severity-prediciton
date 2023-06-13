import os
import argparse
import random
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args

def set_seed(seed=42):
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    # scikit learn use numpy random generator

def save_hist(hist, summary_filepath):
    hist_df = pd.DataFrame(hist.history) 
    with open(os.path.join(summary_filepath,"hist.csv"), mode='w') as f:
        pd.DataFrame(hist.history).to_csv(f)

def plot_history(history,summary_filepath):
    metrics = ['loss', 'auc', 'ap']
    plt.figure(figsize=(15, 10))
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(1,3,n+1)
        plt.plot(history.epoch, history.history[metric], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric], linestyle="--", label='Val')
        plt.plot(history.epoch, history.history['val2_'+metric], linestyle="--", label='Val2')
        plt.plot(history.epoch, history.history['val3_'+metric], linestyle="--", label='Val3')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        plt.legend()
        plt.savefig(os.path.join(summary_filepath,"fig.pdf"),bbox_inches='tight')