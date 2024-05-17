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
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

class MackeyGlass(Dataset):
    """ Dataset for the Mackey-Glass task.
    """
    def __init__(self,
                 tau,
                 constant_past,
                 nmg = 10,
                 beta = 0.2,
                 gamma = 0.1,
                 dt=1.0,
                 splits=(8000., 2000.),
                 start_offset=0.,
                 seed_id=0,
    ):
        """
        Initializes the Mackey-Glass dataset.

        Args:
            tau (float): parameter of the Mackey-Glass equation
            constant_past (float): initial condition for the solver
            nmg (float): parameter of the Mackey-Glass equation
            beta (float): parameter of the Mackey-Glass equation
            gamma (float): parameter of the Mackey-Glass equation
            dt (float): time step length for sampling data
            splits (tuple): data split in time units for training and testing data, respectively
            start_offset (float): added offset of the starting point of the time-series, in case of repeating using same function values
            seed_id (int): seed for generating function solution
        """

        super().__init__()

        # Parameters
        self.tau = tau
        self.constant_past = constant_past
        self.nmg = nmg
        self.beta = beta
        self.gamma = gamma
        self.dt = dt

        # Time units for train (user should split out the warmup or validation)
        self.traintime = splits[0]
        # Time units to forecast
        self.testtime = splits[1]

        self.start_offset = start_offset
        self.seed_id = seed_id

        # Total time to simulate the system
        self.maxtime = self.traintime + self.testtime + self.dt

        # Discrete-time versions of the continuous times specified above
        self.traintime_pts = round(self.traintime/self.dt)
        self.testtime_pts = round(self.testtime/self.dt)
        self.maxtime_pts = self.traintime_pts + self.testtime_pts + 1 # eval one past the end

        # Specify the system using the provided parameters
        self.mackeyglass_specification = [ self.beta * y(0,t-self.tau) / (1 + y(0,t-self.tau)**self.nmg) - self.gamma*y(0) ]

        # Generate time-series
        self.generate_data()

        # Generate train/test indices
        self.split_data()


    def generate_data(self):
        """ Generate time-series using the provided parameters of the equation.
        """
        np.random.seed(self.seed_id)

        # Create the equation object based on the settings
        self.DDE = jitcdde_lyap(self.mackeyglass_specification)
        self.DDE.constant_past([self.constant_past])
        self.DDE.step_on_discontinuities()

        ##
        ## Generate data from the Mackey-Glass system
        ##
        self.mackeyglass_soln = torch.zeros((self.maxtime_pts,1),dtype=torch.float64)
        lyaps = torch.zeros((self.maxtime_pts,1),dtype=torch.float64)
        lyaps_weights = torch.zeros((self.maxtime_pts,1),dtype=torch.float64)
        count = 0
        for time in torch.arange(self.DDE.t+self.start_offset, self.DDE.t+self.start_offset+self.maxtime, self.dt,dtype=torch.float64):
            value, lyap, weight = self.DDE.integrate(time.item())
            self.mackeyglass_soln[count,0] = value[0]
            lyaps[count,0] = lyap[0]
            lyaps_weights[count,0] = weight
            count += 1

        # Total variance of the generated Mackey-Glass time-series
        self.total_var=torch.var(self.mackeyglass_soln[:,0], True)

        # Estimate Lyapunov exponent
        self.lyap_exp = ((lyaps.T@lyaps_weights)/lyaps_weights.sum()).item()


    def split_data(self):
        """ Generate training and testing indices.
        """
        self.ind_train = torch.arange(0, self.traintime_pts)
        self.ind_test = torch.arange(self.traintime_pts, self.maxtime_pts-1)

    def __len__(self):
        """ Returns number of samples in dataset.

        Returns:
            int: number of samples in dataset
        """
        return len(self.mackeyglass_soln)-1

    def __getitem__(self, idx):
        """ Getter method for dataset.

        Args:
            idx (int): index of sample to return

        Returns:
            sample (tensor): individual data sample, shape=(timestamps, features)=(1,1)
            target (tensor): corresponding next state of the system, shape=(label,)=(1,)
        """
        sample = torch.unsqueeze(self.mackeyglass_soln[idx, :], dim=0)
        target = self.mackeyglass_soln[idx+1, :]

        return sample, target
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(x.device)
        out, hidden = self.rnn(x, h0)
        #out, _ = self.rnn(x)
        out = self.fc1(out[:, -1, :])
        out = nn.functional.relu(out)
        out = self.fc2(out)
        return out
    
def make_data(forecasting_horizon, test_size, num_bins, batch_size=1):
    #returns train test split with targets in bins
    mg = MackeyGlass(tau=17,
        constant_past=0.9,
        nmg = 10,
        beta = 0.2,
        gamma = 0.1,
        dt=1.0,
        splits=(1000., 0.),
        start_offset=0.,
        seed_id=0,)
    all_data = Subset(mg, mg.ind_train)
    all_data = list(all_data)
    
    X = np.array([point[0] for point in all_data])
    y = np.array([point[1] for point in all_data])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    bin_edges = np.linspace(np.min(y), np.max(y), num_bins - 1)
    y_train = np.digitize(y_train, bin_edges)
    y_test = np.digitize(y_test, bin_edges)
    #X_train = np.digitize(X_train, bin_edges)
    #X_test = np.digitize(X_test, bin_edges)
    train_teacher_temp = [(X_train[i], y_train[i]) for i in range(len(X_train))]
    train_teacher = DataLoader(train_teacher_temp, shuffle=False, batch_size=batch_size)
    test_teacher = [(X_test[i], y_test[i]) for i in range(len(X_test))]
    test_teacher = DataLoader(test_teacher, shuffle=False, batch_size=batch_size)

    X_s = np.array([point[0] for point in all_data[:-forecasting_horizon]])
    y_s = np.array([point[1] for point in all_data[forecasting_horizon:]])
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_s, y_s, test_size=test_size, shuffle=False)
    bin_edges = np.linspace(np.min(y), np.max(y), num_bins - 1)
    y_train_s = np.digitize(y_train_s, bin_edges)
    y_test_s = np.digitize(y_test_s, bin_edges)
    #X_train_s = np.digitize(X_train_s, bin_edges)
    #X_test_s = np.digitize(X_test_s, bin_edges)
    train_student = [(X_train_s[i], y_train_s[i]) for i in range(len(X_train_s))]
    test_student = [(X_test_s[i], y_test_s[i]) for i in range(len(X_test_s))]

    student_helper = train_teacher_temp[forecasting_horizon:]
    diff_train = len(student_helper) - len(train_student)
    #for i in range(diff_train):
        #train_student.insert(0, (-1, -1))
    for i in range(diff_train):
        student_helper.insert(0, (-1, -1))
    
    train_student = DataLoader(train_student, shuffle=False, batch_size=batch_size)
    test_student = DataLoader(test_student, shuffle=False, batch_size=batch_size)
    student_helper = DataLoader(student_helper, shuffle=False, batch_size=batch_size)
    # Plotting original vs. binned target values
    '''
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(y_s)
    plt.title('Original Target Values')
    plt.xlabel('Time')
    plt.ylabel('Original Value')

    plt.subplot(1, 2, 2)
    plt.plot(y_train_s)
    plt.title('Binned Target Values')
    plt.xlabel('Time')
    plt.ylabel('Bin Number')
    plt.legend()

    plt.tight_layout()
    plt.show()
    '''
    return train_teacher, test_teacher, train_student, test_student, student_helper