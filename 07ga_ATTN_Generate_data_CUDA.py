'''
Generates the data from a trained attention model, iterates across a range of vehicles
Variables:
-range of vehicles
-Data type
-random sample number range
'''
import sys
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
from tqdm import tqdm

from pl_models.base_models_29NOV23 import LSTM, LSTM_sparse, split_data
from pl_models.base_models_29NOV23 import split_data2, add_one, tensorize_list_of_tensors

import functools
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

torch.manual_seed(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

start_vics = 0
end_vics = 12
start_workers = 10
num_workers = 5

data_length = 129600
attn_train_len = 10800

num_epochs = 200
input_dim = 38
hidden_dim = 3
num_layers = 3
output_dim = 38
test_size_fact = 0
lookback = 1200
targets_dim = 4


prediction_length = 1800

ref_range = 0 #where to start the prediction

data_folder = "data/"
model_folder = "models_train/"
folder_true_data = "vehicle_data/"
randomlist_folder_eval = "random_lists/"

if not os.path.isdir(data_folder): 
    os.mkdir(data_folder) 

if not os.path.isdir(model_folder): 
    os.mkdir(model_folder)
    
for vic_name in tqdm(range(start_vics, end_vics)):
    print("On vic: ", vic_name)
    #############Pull Random List Data for Eval########### 
    try:
        file_name = open(randomlist_folder_eval + "uniform_random_start_list_type_v{}_tl{}_pl{}.p".format(vic_name, data_length, prediction_length), 'rb')
        randomlist_eval = pickle.load(file_name)
        file_name.close()
    except:
        print(vic_name, " has no random list")
        continue

    #####################################################
    try:
        file = "adjusted_fault_features_combined_one_label_{}.p".format(vic_name)
        file_name = open(folder_true_data + file, 'rb')
        output_data = pickle.load(file_name)
        file_name.close()
    except:
        continue
    ##################################################### 

    for worker in range(start_workers, start_workers + num_workers):
        try:
            attn_start = randomlist_eval[worker]
        except:
            print("issue with rs worker: ", worker )
            continue
            
        print("On example: ", worker, " with start ", attn_start)
        
        sparse_attn_trained_model_name = model_folder + '06ga_Attn_model_s{}_lk{}_ep{}.pt'.format(attn_start, lookback, num_epochs)

        
        vic_sn, all_data = output_data
        data_features = all_data[attn_start + attn_train_len - lookback:attn_start + attn_train_len + 1, 1:-targets_dim]
        print("data_features.shape", data_features.shape)

        if data_features.shape[0] == 0:
            print("incorrect shape")
            continue
            
        scaler = StandardScaler()
        scaler.fit(data_features)
        data_features_scaled = scaler.transform(data_features)
        try:
            x_data, y_data = split_data2(data_features_scaled, lookback)
        except:
            print("issue with split data on", worker, vic_name)
            continue

        base_data = x_data[ref_range]#this just grabs the first data point with all the previous data, only before time to predict from
        in_data = add_one(base_data)


        model_gen = LSTM_sparse(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, 
                            num_layers=num_layers, window_size = lookback-1).to(device)

        try: 
            model_gen.load_state_dict(torch.load(sparse_attn_trained_model_name, map_location=torch.device(device)))
        except: 
            print("cannot load model")
            continue

        if torch.cuda.is_available():
            in_data = in_data.to(device)
            model_gen = model_gen.to(device)


        out_data = []
        for i in tqdm(range(prediction_length)):
            y_pred,  _, _ = model_gen(in_data)
            out_data.append(y_pred)
            y_pred = add_one(y_pred)
            cat_data = torch.cat((y_pred, in_data), 1)
            in_data = cat_data[:,1:,:]
        out_data = tensorize_list_of_tensors(out_data)
        out_data = out_data.view((prediction_length, input_dim))

        file_name = data_folder + "07ga_Generated_data_v{}_s{}_lk{}_ep{}_pl{}.p".format(vic_name, attn_start, lookback, num_epochs, prediction_length)
        with open(file_name, 'wb') as f: #each file will have one vehicles base data in it
            pickle.dump(out_data, f)
