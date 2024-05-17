import numpy as np
import torch
import math
from jitcdde import jitcdde, y, t, jitcdde_lyap
from torch.utils.data import Dataset
from torch.utils.data import Subset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import RNN, make_data
from tqdm import tqdm
import csv
import torch.nn.functional as F


def main(forecasting_horizon):
    num_bins = 25 # this is how lenient we are with the network prediction
    input_size = 1  # one value 
    hidden_size = 128 # hidden neurons in RNN
    output_size = num_bins # output neurons = num_bins
    num_layers = 1 # RNN layers
    batch_size = 1 # must be 1 for continual learning!!!
    learning_rate = 0.001 

    num_epochs = 15
    teacher_epochs = 20
    
    torch.cuda.empty_cache()
    train_teacher, test_teacher, train_student, test_student, student_helper = make_data(forecasting_horizon, 0.25, num_bins, batch_size)
 
    # Instantiate the models
    teacher = RNN(input_size, hidden_size, output_size, num_layers).to('cuda')
    student = RNN(input_size, hidden_size, output_size, num_layers).to('cuda')
    baseline = RNN(input_size, hidden_size, output_size, num_layers).to('cuda')

    # Loss and optimizer
    optimizer_teacher = optim.Adam(teacher.parameters(), lr=learning_rate)
    optimizer_student = optim.Adam(student.parameters(), lr=learning_rate)
    optimizer_baseline = optim.Adam(baseline.parameters(), lr=learning_rate)
    l1loss = nn.L1Loss()
    teacher.train()
    celoss = nn.CrossEntropyLoss()
    for epoch in range(teacher_epochs):
        total_loss = 0
        for inputs, targets in train_teacher:
            inputs = inputs.float().to('cuda')
            targets = targets.long().to('cuda')
            outputs = teacher(inputs)
            loss = celoss(outputs, targets[0])
            optimizer_teacher.zero_grad()
            loss.backward()
            optimizer_teacher.step()
            total_loss += loss.item()
        print(f'Epoch [{epoch+1}/{teacher_epochs}], Loss: {total_loss/len(train_teacher):.4f}')


    teacher.eval()
    with torch.no_grad():
        total_loss = 0
        predictions_t = []
        true_values_t = []
        softmax_probs_t = []
        correct = 0
        for inputs, targets in test_teacher:
            inputs = inputs.float().to('cuda')
            targets = targets.float().to('cuda')
            outputs = teacher(inputs)
            probabilities = nn.functional.softmax(outputs.detach(), dim=1)
            softmax_probs_t.append(probabilities.flatten().cpu().numpy())
            _, predicted_class = probabilities.max(1)
            #loss = celoss(outputs, targets[0])
            loss = l1loss(predicted_class, targets[0])
            total_loss += loss.item()
            predictions_t.append(predicted_class.cpu().numpy())
            true_values_t.append(targets[0].cpu().detach().numpy())
            correct += (predicted_class == targets[0]).sum().item()
        t_accuracy = correct / len(test_teacher)
        t_loss = total_loss/len(test_teacher)
        print(f'Teacher Test Loss: {t_loss:.4f}, Accuracy: {t_accuracy*100:.2f}%')
    
    # Plotting predictions, true values, and softmax probabilities
    '''
    for i in range(len(predictions_t)):
        plt.figure(figsize=(15, 6))
        plt.plot(softmax_probs_t[i], label='Softmax Probs', color='blue')
        plt.axvline(x=true_values_t[i], color='red', linestyle='--', label='True Value')
        plt.title(f'Sample {i+1} - Softmax Probabilities vs. True Value')
        plt.xlabel('Class')
        plt.ylabel('Probability')
        plt.legend()
        plt.show()
    '''

    plt.figure(figsize=(15, 6))
    plt.subplot(1, 3, 1)
    plt.plot(predictions_t, label='Predictions', color='blue')
    plt.plot(true_values_t, label='True Values', color='red')
    plt.title('Teacher - Predictions vs. True Values')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    
    kl = nn.KLDivLoss(reduction='batchmean')
    alpha = 0.0
    T = 2
    student.train()
    for epoch in range(num_epochs):
        loss_1_list = []
        loss_2_list = []
        for i, ((inputs_teacher, targets_teacher), (inputs_student, targets_student), (inputs_helper, targets_helper)) in enumerate(zip(train_teacher, train_student, student_helper)): 
            inputs_teacher = inputs_teacher.float().to('cuda')
            targets_teacher = targets_teacher.long().to('cuda')
            inputs_helper = inputs_helper.float().to('cuda')
            targets_helper = targets_helper.long().to('cuda')
            targets_student = targets_student.long().to('cuda')
            inputs_student = inputs_student.float().to('cuda')
            if not torch.any(inputs_student == -1.0):
                outputs_student = student(inputs_student)
                teacher.eval()
                with torch.no_grad():
                    assert torch.equal(targets_helper, targets_student)
                    outputs_teacher = teacher(inputs_helper)
                loss_1 = alpha * celoss(outputs_student, targets_student[0])
                loss_2 = (1-alpha)*kl(F.log_softmax(outputs_student / T, dim=1), F.softmax(outputs_teacher / T, dim=1)) * T**2
                loss_1_list.append(loss_1)
                loss_2_list.append(loss_2)
                loss_student = loss_1 + loss_2
                optimizer_student.zero_grad()
                loss_student.backward()
                optimizer_student.step()
        print(f"Average loss of KD for epoch {epoch}: {sum(loss_2_list)/len(loss_2_list):.4f}")
        print(f"Average loss of CE for epoch {epoch}: {sum(loss_1_list)/len(loss_1_list):.4f}")

    student.eval()
    with torch.no_grad():
        total_loss = 0
        predictions_s = []
        true_values_s = []
        correct = 0
        for inputs, targets in test_student:
            inputs = inputs.float().to('cuda')
            targets = targets.float().to('cuda')
            outputs = student(inputs)
            probabilities = nn.functional.softmax(outputs.detach(), dim=1)
            _, predicted_class = probabilities.max(1)

            #loss = celoss(outputs, targets[0])
            loss = l1loss(predicted_class, targets[0])
            total_loss += loss.item()

            predictions_s.append(predicted_class.cpu().numpy())
            true_values_s.append(targets[0].cpu().detach().numpy())
            correct += (predicted_class == targets[0]).sum().item()
        s_accuracy = correct / len(test_student)
        s_loss = total_loss/len(test_student)
        print(f'Student Test Loss: {s_loss:.4f}, Accuracy: {s_accuracy*100:.2f}%')

    plt.subplot(1, 3, 2)
    plt.plot(predictions_s, label='Predictions', color='blue')
    plt.plot(true_values_s, label='True Values', color='red')
    plt.title('Student - Predictions vs. True Values')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()

    ###Baseline###
    baseline.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in train_student:
            if not torch.any(inputs == -1.0):
                inputs = inputs.float().to('cuda')
                targets = targets.long().to('cuda')
                outputs = baseline(inputs)
                loss = celoss(outputs, targets[0])
                optimizer_baseline.zero_grad()
                loss.backward()
                optimizer_baseline.step()
                total_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/(len(train_student)-1):.4f}')


    baseline.eval()
    with torch.no_grad():
        total_loss = 0
        predictions_b = []
        true_values_b = []
        correct = 0
        for inputs, targets in test_student:
            inputs = inputs.float().to('cuda')
            targets = targets.float().to('cuda')
            outputs = baseline(inputs)
            probabilities = nn.functional.softmax(outputs.detach(), dim=1)
            _, predicted_class = probabilities.max(1)

            #loss = celoss(outputs, targets[0])
            loss = l1loss(predicted_class, targets[0])
            total_loss += loss.item()
    
            predictions_b.append(predicted_class.cpu().numpy())
            true_values_b.append(targets[0].cpu().detach().numpy())
            correct += (predicted_class == targets[0]).sum().item()
        b_accuracy = correct / len(test_student)
        b_loss = total_loss/len(test_student)
        print(f'Baseline Test Loss: {b_loss:.4f}, Accuracy: {b_accuracy*100:.2f}%')
    
    plt.subplot(1, 3, 3)
    plt.plot(predictions_b, label='Predictions', color='blue')
    plt.plot(true_values_b, label='True Values', color='red')
    plt.title('Baseline - Predictions vs. True Values')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.show()  
    return [t_loss, s_loss, b_loss]

horizon = 75
main(horizon)
'''
horizons = [5, 15, 25, 50, 75, 100, 125, 150]
results = dict()
for horizon in horizons:
    results[horizon] = []
    for i in range(0, 5):
        results[horizon].append(main(horizon))
    print(results[horizon])
print(results)
'''