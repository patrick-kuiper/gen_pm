'''
Generates trains the attention model to generate the features
Variables:
-range of vehicles
-Data type
-random sample number range
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import math, time
import itertools
from datetime import datetime
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import pickle
import numpy as np
import pandas as pd
import os
from pylab import mpl, plt
import time
from collections import defaultdict
from sparsemax import Sparsemax
import sys

from pl_models.base_models_29NOV23 import LSTM, LSTM_sparse, split_data, tensorize_list_of_tensors

import functools

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

torch.manual_seed(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

deepar_train_start = 0
data_length = 129600
attn_train_len = 10800

start_vics = 0
num_vics = 13
start_workers = 10
num_workers = 50

targets_dim = 4
input_dim = 38
hidden_dim = 3
num_layers = 3
output_dim = 38
test_size_fact = 0
prediction_length = 1800
lookback = 1200
num_epochs = 200

data_folder = "data/"
model_folder = "models_train/"
folder_true_data = "vehicle_data/"
randomlist_folder = "random_lists/"

if not os.path.isdir(data_folder): 
    os.mkdir(data_folder) 

if not os.path.isdir(model_folder): 
    os.mkdir(model_folder)
    
for vic_name in range(start_vics, start_vics + num_vics):
    try:
        file_name = open(randomlist_folder + "uniform_random_start_list_type_v{}_tl{}_pl{}.p".format(vic_name, data_length, prediction_length), 'rb')
        randomlist = pickle.load(file_name)
        file_name.close()
    except:
        print(vic_name, " has no random list")
        continue

    #############Pull True Vehicle Data########### 
    file_true_data = "adjusted_fault_features_combined_one_label_{}.p".format(vic_name)
    file_name_true_data = open(folder_true_data + file_true_data, 'rb')
    output_data = pickle.load(file_name_true_data)
    file_name_true_data.close()
    
    vic_sn, all_data = output_data
    ##################################################### 
    for worker in range(start_workers, start_workers + num_workers):

        try:
#             attn_start, type_of_sample = randomlist[worker]
            attn_start = randomlist[worker]
        except:
            print("no random list")
            continue
        if attn_start < deepar_train_start + data_length: #this makes sure the sample is after the deepar training period
            print(worker, attn_start, " start issue")
            continue

            
        #initiate model and optimizer
        model = LSTM_sparse(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, window_size = lookback - 1)
        criterion = torch.nn.MSELoss(reduction='mean')
        optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
        
        data_features = all_data[attn_start:attn_start + attn_train_len, 1:-targets_dim]
        scaler = StandardScaler()
        scaler.fit(data_features)
        data_features_scaled = scaler.transform(data_features)
        #split data
        try:
            x_train, y_train, x_test, y_test = split_data(data_features_scaled, lookback, test_size_fact)
        except:
            continue
        x_train = torch.from_numpy(x_train).type(torch.Tensor)
        y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)#makes usable by pytorch during training 
        
        if torch.cuda.is_available():
            x_train = x_train.to(device)
            model = model.to(device)
            y_train_lstm = y_train_lstm.to(device)
            
        #train model
        betais = []
        for t in range(num_epochs):
            with open(data_folder + '/06ga_Train_Status_{}_{}.txt'.format(vic_name, worker), 'w') as f:
                f.write("On vehicle: " + str(vic_name) + " on worker " + str(worker) + " on start: " + str(attn_start) + " on epoch: " + str(t))
            print("on epoch: ", t)
            y_train_pred, hidden, betai = model(x_train)
            betais.append(betai)
            loss = criterion(y_train_pred, y_train_lstm)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            torch.save(model.state_dict(), model_folder + '/06ga_Attn_model_s{}_lk{}_ep{}.pt'.format(attn_start, lookback, num_epochs))
        
        #save attn weights
        #Get Last attn wt vector
        betais = tensorize_list_of_tensors(betais).cpu().squeeze().detach().numpy().squeeze()[-1,:,:].squeeze()
        attn_wts_name = '/06ga_attn_wts_s{}_lk{}_ep{}.p'.format(attn_start, lookback, num_epochs)
        with open(model_folder + attn_wts_name, 'wb') as f: #each file will have one vehicles base data in it
            pickle.dump(betais, f)
        print(y_train_pred.shape)
        print(data_features_scaled.shape)
        trace_data = {"y_train_pred": y_train_pred, "data_features_scaled": data_features_scaled, "start": attn_start, "lookback": lookback, "num_epochs": num_epochs, "attn_train_len": attn_train_len}
        #############Make Directories and Pull Data###########

        file_name = data_folder + "06ga_attn_feat_trace_data_s{}_lk{}_ep{}_v{}.p".format(attn_start, lookback, num_epochs, vic_name)
        with open(file_name, 'wb') as f: #each file will have one vehicles base data in it
            pickle.dump(trace_data, f)
            