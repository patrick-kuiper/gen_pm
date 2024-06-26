"""
This will take in the data congolmerated by 06gv and train a VAE and KMeans Model 
**UPDATED TO USE DBSCAN
INPUT:
1. Data Generated by 06gv, seperated in to fault and no-fault time periods.
OUTPUT:
1. VAE model for data compression to latent domain
2. KNN model for determining what "state" vehicle is in and generating stochastically from this state for conditioning
"""

import pandas as pd
import math
from itertools import combinations
from collections import defaultdict
import numpy as np
import sys

import matplotlib.pyplot as plt
import matplotlib.colors
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
from sklearn.cluster import KMeans
from joblib import dump, load

from pl_models.base_models_29NOV23 import CVAE, MyDataset_CVAE, loss_function, test
from pl_models.base_models_29NOV23 import train, make_optimizer
from pl_models.base_models_29NOV23 import VAE, MyDataset_CVAE, test_VAE 
from pl_models.base_models_29NOV23 import train_VAE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    
########## Define parameters and load data #############    
    num_epochs = 100

    num_attributes = 39
    num_features = 38
    num_steps_pred = 1800
    num_labels = 2
    target_ref_col = num_features * num_steps_pred + 1
    num_meta_features = 4

    lr = 0.001
    n_in = (num_features * num_steps_pred)
    n_hid = 400
    z_dim = 3

    view_1 = 180
    view_2 = 180

    optimizer_name = 'Adam'
    test_fact_vae = 0.9
    batch = 32
    seed = 1

    meta_eng_hr_key = 'LVL90.TEST.DLA.DSCSetupInstalledEngineHours'
    meta_fam_mem_key ='LVL90.PLAT.Asset.product_family_member'
    meta_loc_key ='LVL90.TEST.DLA.Location'
    meta_odo_key ='LVL90.TEST.DLA.DSCSetupInstalledOdometer'

    location_dict = {'AFG-Apache': 0,
                     'AFG-Frontenac': 1,
                     'AFG-Kandahar': 2,
                     'AFG-Lindsey': 3,
                     'AFG-Masum Ghar': 4,
                     'AFG-Pacemaker': 5,
                     'AFG-Pasab': 6,
                     'AFG-Zangabad': 7}

    model_dict = {'MODEL0011': 0, 
                  'MODEL0012': 1, 
                  'MODEL0014': 2, 
                  'MODEL0015': 3, 
                  'MODEL0016': 4}

    meta_type_dict = {0: 'Odometer', 
                      1: 'Engine Hours', 
                      2: 'Vehicle Family', 
                      3: 'Location'}
    
########## Load in Data #####################################  
    vae_folder = "vae_data/"
    folder_true_data = "vehicle_data/"
    if not os.path.isdir(vae_folder): 
        os.mkdir(vae_folder)
    device = torch.device(device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    file_name = open(vae_folder + 'test_feature_data.p', 'rb')
    gen_data = pickle.load(file_name)
    file_name.close()

    #load meta data
    file = open(folder_true_data + 'meta-data-22MAR23.p', 'rb')
    meta_data = pickle.load(file)
    file.close()

    #load ref dict for each vic
    file = open(folder_true_data + 'full_vic_ref_id_number', 'rb')
    vic_num_ref = pickle.load(file)
    file.close()
########## Transform Data #####################################       
    #Extract data from dict it was saved in
    ser_to_num_dict = {k[0]:v[0] for (k, v) in vic_num_ref.items()} #get ref dict mappign ser num to vic ref int
    gen_data_keys = list(gen_data.keys())
    data_list = []
    for i, gen_key in enumerate(gen_data_keys):
        if gen_data[gen_key].shape[0] != (num_attributes * num_steps_pred):
            continue
        else:
            #get metadata
            all_meta_data_i = meta_data[ser_to_num_dict[gen_key[0]]]
            odo_i = int(all_meta_data_i[meta_odo_key])
            eng_hr_i = int(all_meta_data_i[meta_eng_hr_key])
            fam_mem_i = model_dict[all_meta_data_i[meta_fam_mem_key]]
            loc_i = location_dict[all_meta_data_i[meta_loc_key]]
            #Below is order of meta-data
            all_data_i = np.array([odo_i, eng_hr_i, fam_mem_i, loc_i]).astype("float32")
            #load all into list
            gen_data_w_meta = np.hstack((gen_data[gen_key], np.array(all_data_i)))
            data_list.append(gen_data_w_meta)

    data_array = np.array(data_list)[:,:-num_meta_features].reshape((len(data_list), 
                                                                                 num_steps_pred, num_attributes)) 
    meta_array = np.array(data_list)[:,-num_meta_features:]
    data, target = data_array[:, :, :-1].reshape((data_array.shape[0], num_features * num_steps_pred)), data_array[:, :, -1]
    index_w_fault = np.where(target.max(1) == 1)[0]
    data_w_target = np.hstack((data, target.max(1).reshape(data.shape[0],-1))).astype("float32") #stash sensor data with fault data for scaling

    #Scale Data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data_w_target[:,:-1])
    data_w_target[:,:-1] = scaler.transform(data_w_target[:,:-1])

    #Put into dataloader for training
    vae_input_data = torch.tensor(data_w_target[:,:-1])
    data_transform = MyDataset_CVAE(vae_input_data)
    train_lim = round(data_transform.x.shape[0] * test_fact_vae)
    trainloader = torch.utils.data.DataLoader(data_transform[:train_lim], batch_size=batch, shuffle=True)
    testloader = torch.utils.data.DataLoader(data_transform[train_lim:], batch_size=batch, shuffle=True)

    
    ########## Train VAE Model #####################################
    vae = VAE(n_in, n_hid, z_dim).to(device)
    optimizer = make_optimizer(optimizer_name, vae, lr=lr)
    for epoch in range(1, num_epochs + 1):
        train_VAE(vae, device, trainloader, optimizer, epoch)
    torch.save(vae.state_dict(), vae_folder + 'VAE_model_eps{}_zdim{}.pt'.format(num_epochs, z_dim))



