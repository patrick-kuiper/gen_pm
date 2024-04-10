'''
This script compiles the data for the CVAE. The samples are taken over randomly sampled periods based on the time legth and if or if not fault diurin ghte generated period. 
**The random sample ranges should be different from the random samples used to evaluate performance to keep with the idea that these are previously generated use profile data that we are generating from
INPUTS:
data_type = {0,1} training for faults or no faults in generated datatime
data_length = length of trained data
OUTPUT:
test_feature_data = dict with the sampled data to train CVAE, keyed on (vic_name, start)
'''

import pandas as pd
import math
from itertools import combinations
from collections import defaultdict
import numpy as np

import matplotlib.pyplot as plt
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.optim as optim
from torch.autograd import Variable
import pickle
from sklearn.preprocessing import StandardScaler

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os
import warnings
warnings.filterwarnings("ignore")

import math as ma
from random import sample

import itertools
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib import style
import pickle


data_length = 129600
prediction_length = 1800

vic_start = 0
vic_end = 12
dp_start = 0
dp_end = 300

fault_type_dict = {"combined": 39, "brakes": 40, "engine": 41, "transmission": 42} 
fault_select = "combined"

label_start = 39
input_dim = 38
hidden_dim = 3
num_layers = 3
output_dim = 38
test_size_fact = 0
lookback = 1200
num_epochs = 200

out_data_folder = "vae_data/"
folder_true_data = "vehicle_data/"
randomlist_folder = "random_lists/"

if not os.path.isdir(out_data_folder): 
    os.mkdir(out_data_folder)

string_errors = []
test_feature_data = {}

for vic_name in range(vic_start, vic_end):
    print("on vic: ", vic_name)
    try:
        file_true_data = "adjusted_fault_features_combined_one_label_{}.p".format(vic_name)
        file_name_true_data = open(folder_true_data + file_true_data, 'rb')
        output_data = pickle.load(file_name_true_data)
        file_name_true_data.close()
    except:
        string_error_1 = "no data for vic: " + str(vic_name)
        print(string_error_1)
        string_errors.append(string_error_1)
        continue
    
    vic_sn, all_data = output_data
    
    try:
        rs_file_name = open(randomlist_folder + "uniform_random_start_list_type_v{}_tl{}_pl{}.p".format(vic_name, data_length, prediction_length), 'rb')
        randomlist_eval = pickle.load(rs_file_name)
        rs_file_name.close()
    except:
        string_error_1 = "no rs data for vic: " + str(vic_name)
        print(string_error_1)
        string_errors.append(string_error_1)
        continue

    for data_point in range(dp_start,dp_end):
        try:
            start = randomlist_eval[data_point]
        except:
            string_error_3 = "no rs issue for vic: " + str(vic_name) + " data point " + str(data_point)
            print(string_error_3)
            string_errors.append(string_error_3)
            continue
        print("on data point: ", data_point)

        t = start+data_length
        T = start+data_length+prediction_length
        
        #Put selected label with features: ADDED 08JAN24
        data_features = all_data[t:T, 1:label_start]
        selected_fault_col_num = fault_type_dict[fault_select]
        data_target = all_data[t:T, selected_fault_col_num].reshape(-1,1)  
        data_features_targets = np.hstack((data_features, data_target))
        
        if data_features_targets.shape[0] == 0:
            string_error_4 = "issue wiht data shape for vic: " + str(vic_name) + " data point " + str(data_point)
            print(string_error_4)
            string_errors.append(string_error_4)
            continue
        if np.isnan(data_features_targets.astype(np.float32)).sum() > 0:
            string_error_5 = "issue with nan for vic: " + str(vic_name) + " data point " + str(data_point)
            print(string_error_5)
            string_errors.append(string_error_5)
            continue
        print(data_features_targets.shape[0])
        features_flat_out = data_features_targets.flatten()
        
        test_feature_data[(vic_name, start)] = features_flat_out
        

    file_name = out_data_folder + 'error_data.p'
    with open(file_name, 'wb') as f: #each file will have one vehicles base data in it
        pickle.dump(string_errors, f)        

file_name = out_data_folder + 'test_feature_data.p'
with open(file_name, 'wb') as f: #each file will have one vehicles base data in it
    pickle.dump(test_feature_data, f)