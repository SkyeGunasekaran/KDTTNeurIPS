import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.pool2 = nn.AvgPool3d(kernel_size=3, stride=2)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        # LSTM module
        self.lstm = nn.LSTM(input_size=32*60, hidden_size=512, num_layers=3, batch_first=True)
        
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
        if verbose: print('shape after pool1:', x.shape)

        x = self.bn2(x)
        x = F.relu(self.conv2(x))
        if verbose: print('shape after conv2:', x.shape)
        x = self.pool2(x)
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