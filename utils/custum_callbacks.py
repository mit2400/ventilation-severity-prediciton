import tensorflow as tf

class MultiValidationCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_datasets):
        super().__init__()
        self.val_datasets = val_datasets
    def on_epoch_end(self, epoch, logs=None):
        metrics = [[] for _ in range(3)]
        for i, val_dataset in enumerate(self.val_datasets):
            val_loss, val_auc, val_ap = self.model.evaluate(val_dataset,verbose=0)
            logs[f'val{i+2}_loss'] = val_loss
            logs[f'val{i+2}_auc'] = val_auc
            logs[f'val{i+2}_ap'] = val_ap
        formatted_items = [f'{key}: {value:.3f}' for key, value in logs.items()]
        print(f"Epoch{epoch}\t", end='')
        print(formatted_items)