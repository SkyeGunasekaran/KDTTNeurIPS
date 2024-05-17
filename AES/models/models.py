import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

class EarlyStopping:
    def __init__(self, patience=6, mode='min'):
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

class CNN_LSTM_Model(nn.Module):
    def __init__(self, X_train_shape):
        super(CNN_LSTM_Model, self).__init__()
        
        # CNN module
        self.bn1 = nn.BatchNorm3d(num_features=X_train_shape[1])
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, 
                               kernel_size=(int(X_train_shape[2]/2), 5, 5), stride=(1, 2, 2))
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.bn2 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.pool2 = nn.AvgPool3d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        # LSTM module
        self.lstm = nn.LSTM(input_size=int(X_train_shape[2]/12)*32*66, hidden_size=512, num_layers=3, batch_first=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 2)  # 2 output classes
        
    def forward(self, x, verbose=False):
        # CNN forward pass
        if verbose: print('input size', x.shape)
        x = self.bn1(x)
        x = F.relu(self.conv1(x))
        if verbose: print('shape after conv1:', x.shape)
        x = self.pool1(x)
        #x = self.dropout1(x)
        if verbose: print('shape after pool1:', x.shape)

        x = self.bn2(x)
        x = F.relu(self.conv2(x))
        if verbose: print('shape after conv2:', x.shape)
        x = self.pool2(x)
        #x = self.dropout2(x)
        if verbose: print('shape after pool2:', x.shape)

        # Reshape for LSTM
        x = x.squeeze(2)
        x = self.flatten(x)
        if verbose: print('shape after flatten:', x.shape)
        # LSTM forward pass

        x, _ = self.lstm(x)
        if verbose: print('shape after LSTM:', x.shape)
        
        # Fully connected layers
        x = self.fc1(x)  # Select the last time step's output
        if verbose: print('shape after fc1:', x.shape)
        x = self.dropout(x)
        x = self.fc2(x)
        if verbose: print('shape after fc2:', x.shape)
        return x
    
class CNN_Model(nn.Module):
    def __init__(self, X_train_shape):
        super(CNN_Model, self).__init__()
        
        # CNN module
        self.bn1 = nn.BatchNorm3d(num_features=X_train_shape[1])
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(int(X_train_shape[2]/2), 5, 5), stride=(1, 2, 2))
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.bn2 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.pool2 = nn.AvgPool3d(kernel_size=2, stride=2)

        self.bn3 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        
        # Fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 2)  # 2 output classes
        
    def forward(self, x, verbose=False):
        # CNN forward pass

        if verbose: print('input size', x.shape)
        x = self.bn1(x)
        x = F.relu(self.conv1(x))
        if verbose: print('shape after conv1:', x.shape)
        x = self.pool1(x)
        if verbose: print('shape after pool1:', x.shape)

        x = self.bn2(x)
        x = F.relu(self.conv2(x))
        if verbose: print('shape after conv2:', x.shape)
        x = self.pool2(x)
        #x = self.dropout2(x)
        if verbose: print('shape after pool2:', x.shape)

        # Reshape for Conv2d
        x = x.squeeze(2)

        x = self.bn3(x)
        x = F.relu(self.conv3(x))
        if verbose: print('shape after conv3:', x.shape)
        x = self.pool3(x)
        if verbose: print('shape after pool3:', x.shape)

        x = self.flatten(x)
        if verbose: print('shape after flatten:', x.shape)
        
        x = self.fc1(x) 
        if verbose: print('shape after fc1:', x.shape)
        x = self.dropout(x)
        x = self.fc2(x)
        if verbose: print('shape after fc2:', x.shape)
        return x