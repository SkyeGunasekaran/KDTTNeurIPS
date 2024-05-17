import json
import os.path
from utils.log import log
from utils.load_signals_student import PrepDataStudent
from utils.prep_data_student import train_val_test_split_continual_s
from models.models import CNN_LSTM_Model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import optuna 


def main(target, distillation_method):
    torch.cuda.empty_cache()
    parameters = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    trials = 5
    epochs = 25
    T = 20

    with open('student_settings.json') as k:
        student_settings = json.load(k)
    
    if 'Dog_' in target:
        model_path = 'teacher_dog'
    elif target == 'Patient_1' or target == 'Patient_2':
        model_path = target

    path = f'{model_path}.pth'
    teacher = torch.load(path)
    teacher.eval()

    ictal_X, ictal_y = PrepDataStudent(target, type='ictal', settings=student_settings).apply()
    interictal_X, interictal_y = PrepDataStudent(target, type='interictal', settings=student_settings).apply()
    X_train, y_train, X_test, y_test = train_val_test_split_continual_s(ictal_X, ictal_y, interictal_X, interictal_y, 0.35)   
    
    Y_train       = torch.tensor(y_train).type(torch.LongTensor).to('cuda')
    X_train       = torch.tensor(X_train).type(torch.FloatTensor).to('cuda')
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=False)
    param_dict = dict()
    for param in parameters:
        alpha, beta = param, 1-param
        param_list = []
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
                    with torch.no_grad():
                        teacher_logits = teacher(X_batch)
                    student_logits = student(X_batch)
                    if distillation_method == 'KL':
                        soft_targets = F.softmax(teacher_logits / T, dim=1)
                        soft_prob = F.log_softmax(student_logits / T, dim=1)
                        soft_targets_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (T**2) 
                    if distillation_method == 'MSE':
                        soft_targets_loss = F.mse_loss(student_logits, teacher_logits)
                    #print('Student Loss, Distillation Loss: ', alpha*ce_loss(student_logits, Y_batch), beta*soft_targets_loss)
                    loss = alpha*ce_loss(student_logits, Y_batch) + beta*soft_targets_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                #print(f'Loss for epoch {epoch}: {total_loss/len(train_loader)}')
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
            param_list.append(auc_test)
        param_dict[param] = param_list

    return param_dict

if __name__ == '__main__':
    targets = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
    targets = ['Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
    for target in targets:
        target_results = main(distillation_method='KL', target=target)
        print(target_results)
        with open('results.txt', 'a') as f:
            f.write(f'{target}_results=\n')
            f.write(str(target_results))
            f.close()