import os
import json
import argparse
import random
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser(description='training arguments')
    # parser.add_argument('--testcode', action='store_true', help='run test code')
    parser.add_argument('--mode', type=str, default='train', help='[train,eval.code_test] train is default')
    parser.add_argument('--search_params', action='store_true', help='do hyperparmeter search')
    parser.add_argument('--config_path', type=str, default='./configs/base.json', help='load configs from given path')
    parser.add_argument('--config_path', type=str, default='./configs/base.json', help='load configs from given path')
    #add arguments todo
    args = parser.parse_args()
    return args

def get_configs(args):
    if args.mode == 'code_test':
        with open('./configs/base.json', 'r') as f:
            configs = json.load(f)
        summary_filepath = f"./logs/test"
        print(f"Loading test configs")
    elif args.search_params:
        #todo
        pass
    else:
        with open(args.config_path, 'r') as f:
            configs = json.load(f)
        #file_paths
        # _configs = '_'.join(str(value) for value in configs.values())
        _configs = '_'.join(f'{key}{value}' for key, value in configs.items())
        summary_filepath = f"./logs/model_{_configs}"
        print(f"Loading configs from {args.config_path}")

    return configs, summary_filepath

def set_seed(seed=42):
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    # scikit learn use numpy random generator

def save_hist(hist, summary_filepath):
    hist_df = pd.DataFrame(hist.history) 
    with open(os.path.join(summary_filepath,"hist.csv"), mode='w') as f:
        pd.DataFrame(hist.history).to_csv(f)

def save_config(config, summary_filepath):
    config_json = json.dumps(config, indent=4)
    with open(os.path.join(summary_filepath,'config.json'), 'w') as f:
        f.write(config_json)

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