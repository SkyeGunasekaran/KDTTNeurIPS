import json
import csv
import os
import os.path
import numpy as np
from utils.load_signals_student import PrepDataStudent
from utils.load_signals_teacher import PrepDataTeacher
from utils.prep_data_student import train_val_test_split_continual_s
from utils.prep_data_teacher import train_val_test_split_continual_t
from models.models import CNN_LSTM_Model
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

def makedirs(dir):
    try:
        os.makedirs(dir)
    except:
        pass
def find_change_indices(lst):
    change_indices = []
    prev_value = lst[0]
    
    for i, value in enumerate(lst[1:], start=1):
        if value != prev_value:
            change_indices.append(i)
            prev_value = value
    
    print(change_indices)

def main(target):
    torch.cuda.empty_cache()
    trials = 1
    epochs=25
    
    with open('student_settings.json') as k:
        student_settings = json.load(k)

    makedirs(str(student_settings['cachedir']))
  
    student_results = []
    
    ictal_X, ictal_y = PrepDataStudent(target, type='ictal', settings=student_settings).apply()
    interictal_X, interictal_y = PrepDataStudent(target, type='interictal', settings=student_settings).apply()
    X_train, y_train, X_test, y_test = train_val_test_split_continual_s(ictal_X, ictal_y, interictal_X, interictal_y, 0.35)  
    
    Y_train       = torch.tensor(y_train).type(torch.LongTensor).to('cuda')
    X_train       = torch.tensor(X_train).type(torch.FloatTensor).to('cuda')
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=False)
    for i in range(trials):
        student = CNN_LSTM_Model(X_train.shape).to('cuda')
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
            print(f'Loss for epoch {epoch}: {total_loss/len(train_loader)}')
            pbar.update(1)
        pbar.close() 

        student.eval()
        X_tensor = torch.tensor(X_test).float().to('cuda')
        y_tensor = torch.tensor(y_test).long().to('cuda')
        with torch.no_grad():
            predictions = student(X_tensor)
        predictions = F.softmax(predictions, dim=1)
        predictions = predictions[:, 1].cpu().numpy()
        auc_test = roc_auc_score(y_tensor.cpu(), predictions)
        print('Test AUC is:', auc_test)
        student_results.append(auc_test)
    
    print(f'Results for patient {target}', student_results)
    return student_results

if __name__ == '__main__':
    targets = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
    results = dict()
    for i in targets:
        results[i] = main(target=i)
    print(results)
    