import json
import numpy as np
from utils.load_signals_student import PrepDataStudent
from utils.prep_data_student import train_val_test_split_continual_s
from models.model import CNN_LSTM_Model
from os import makedirs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

def main():
    print ('Main - Seizure Prediction')
    torch.cuda.empty_cache()

    epochs=25
    
    with open('student_settings.json') as k:
        student_settings = json.load(k)
    #makedirs(str(student_settings['cachedir']))

    student_results = []
    targets = ['1','2','3','5','9','10','13','18','19','20','21','23']

    for target in targets:
        student_losses = []
        ictal_X, ictal_y = PrepDataStudent(target, type='ictal', settings=student_settings).apply()
        interictal_X, interictal_y = PrepDataStudent(target, type='interictal', settings=student_settings).apply()
        X_train, y_train, X_test, y_test = train_val_test_split_continual_s(ictal_X, ictal_y, interictal_X, interictal_y, 0.35)
        student = CNN_LSTM_Model(X_train.shape).to('cuda')

        Y_train       = torch.tensor(y_train).type(torch.LongTensor).to('cuda')
        X_train       = torch.tensor(X_train).type(torch.FloatTensor).to('cuda')
        train_dataset = TensorDataset(X_train, Y_train)
        train_loader  = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)

        ce_loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(student.parameters(), lr=5e-4, betas=(0.9,0.999), eps=1e-8)

        pbar = tqdm(total=epochs)
        for epoch in range(epochs):
            student.train()
            total_loss = 0
            for X_batch, Y_batch in train_loader:
                X_batch, Y_batch = X_batch.to("cuda"), Y_batch.to("cuda")
                student_logits = student(X_batch)
                loss = ce_loss(student_logits, Y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            pbar.update(1)
            student_losses.append(total_loss / len(train_loader))
        pbar.close()   

        student.eval()
        student.to('cuda')
        X_tensor = torch.tensor(X_test).float().to('cuda')
        y_tensor = torch.tensor(y_test).long().to('cuda')

        with torch.no_grad():
            predictions = student(X_tensor)
        predictions = F.softmax(predictions, dim=1)
        
        predictions = predictions[:, 1].cpu().numpy()
        auc_test = roc_auc_score(y_tensor.cpu(), predictions)
        print('Test AUC is:', auc_test)
        student_results.append(auc_test)
        #path = f'pytorch_models/Patient_{target}_prediction'
        #torch.save(student, path)
        plt.plot(student_losses, label='predictor loss')
        plt.show()
    print(student_results)
if __name__ == "__main__":
    main()



