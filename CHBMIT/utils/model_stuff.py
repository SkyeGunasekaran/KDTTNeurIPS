import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix

def training_interpretability(model, loader):
    for X_batch, y_batch in loader:
        X_tensor, y_tensor = X_batch.to("cuda"), y_batch.to("cuda")
    with torch.no_grad():
        predictions = model(X_tensor)
    predictions = F.softmax(predictions)
    predictions = predictions[:, 1].cpu().numpy()
    auc_test = roc_auc_score(y_tensor.cpu(), predictions)
    binary = np.where(predictions > 0.5, 1, 0)
    cm = confusion_matrix(y_tensor.cpu(), binary)
    print('Train AUC is:', auc_test)
    print('confusion matrix', cm)

class EarlyStopping:
    def __init__(self, patience=5, mode='min'):
        """
        Initialize the early stopping mechanism.

        :param patience: Number of epochs with no improvement to wait before stopping training.
        :param mode: 'min' to stop when the monitored loss doesn't decrease, 'max' if it doesn't increase.
        """
        self.patience = patience
        self.mode = mode
        self.best_loss = float('inf') if mode == 'min' else -float('inf')
        self.best_epoch = 0
        self.wait = 0
        self.stopped = False

    def step(self, loss, epoch):
        """
        Update the early stopping mechanism with the current loss and epoch.

        :param loss: The current loss value.
        :param epoch: The current epoch number.
        """
        if (self.mode == 'min' and loss < self.best_loss) or (self.mode == 'max' and loss > self.best_loss):
            self.best_loss = loss
            self.best_epoch = epoch
            self.wait = 0  # Reset wait counter as the loss improved
        else:
            self.wait += 1  # Increment wait counter

        # Check if we need to stop training
        if self.wait >= self.patience:
            self.stopped = True

    def is_stopped(self):
        """
        Return whether training should be stopped.

        :return: True if training should be stopped, otherwise False.
        """
        return self.stopped

