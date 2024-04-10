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

########## Define parameters and load data #############    
num_epochs = 100
eps_db = 0.25 #0.75
min_samples_db = 20 #35

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

num_cluster = 5

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

vae_load = VAE(n_in, n_hid, z_dim).to(device)
vae_load.load_state_dict(torch.load(vae_folder + 'VAE_model_eps{}_zdim{}.pt'.format(num_epochs, z_dim), map_location=torch.device(device)))

########## Train KMEANS Model #####################################
num_labels = 2
features, target = data_w_target[:,:-1], data_w_target[:,-1]
target = torch.tensor(target).to(torch.int64)
target = torch.nn.functional.one_hot(target, num_labels).to(device)#new
features = torch.tensor(features).to(device)
output, mu, logvar, z = vae_load(features)

labels = data_w_target[:,-1]
z_in = z.cpu().detach().numpy()
kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(z_in)
pred_cluster_labels = kmeans.predict(z_in)
labels_kmeans = kmeans.labels_
list_cluster_labels = list(set(pred_cluster_labels))
kmeans_dist = [[lbl, pred_cluster_labels[pred_cluster_labels == lbl].shape[0] / pred_cluster_labels.shape[0]] for lbl in list_cluster_labels]
n_clusters_ = len(pred_cluster_labels)

#save model
with open(vae_folder + 'Kmeans_model_eps{}_zdim{}.p'.format(num_epochs, z_dim), 'wb') as f: 
        pickle.dump(kmeans, f)
        

percent_fault_dict_by_state = {i: (np.sum((labels_kmeans == i) & (data_w_target[:,-1] == 1)) / np.sum(labels_kmeans == i)) for i in range(num_cluster)}


print(percent_fault_dict_by_state)

with open(vae_folder + 'percent_fault_dict_by_state_eps{}_zdim{}.p'.format(num_epochs, z_dim), 'wb') as f: 
        pickle.dump(percent_fault_dict_by_state, f)




########## Plot Clustering States #####################################
#Plot Fault Data
keep_labels_fault = data_w_target[:,-1]
fig = plt.figure(figsize=(10, 10))
cmap = matplotlib.colors.ListedColormap(['green', 'red'])#plt.cm.jet
cmaplist = [cmap(i) for i in range(cmap.N)]
ax = plt.axes(projection ='3d')
# ax.scatter(keep_z[:,0].cpu().detach().numpy(), keep_z[:,1].cpu().detach().numpy(), keep_z[:,2].cpu().detach().numpy(),c=keep_labels_fault, cmap=cmap)
ax.scatter(z_in[:,0], z_in[:,1], z_in[:,2],c=keep_labels_fault, cmap=cmap)
ax.set_title('Sensor Data in Latent Space with Fault', fontsize = 20)
ax.view_init(view_1, view_2)
scalarmappaple = cm.ScalarMappable(cmap=cmap)
scalarmappaple.set_array(keep_labels_fault)
cbar = fig.colorbar(scalarmappaple, shrink=0.6, ticks=[0, 1])
cbar.ax.set_yticklabels(["No Fault", "Fault"], fontsize = 15)
plt.savefig(vae_folder + "Kmeans_all_data_fault_{}.png".format(num_epochs), bbox_inches="tight")

#Cluster Data
fig = plt.figure(figsize=(10, 10))
# cmap = plt.cm.jet
cmap = matplotlib.colors.ListedColormap(['green', 'red', "blue", "purple", "orange"])
cmaplist = [cmap(i) for i in range(cmap.N)]
ax = plt.axes(projection ='3d')
ax.scatter(z_in[:,0], z_in[:,1], z_in[:,2],c=labels_kmeans, cmap=cmap)
ax.set_title('Sensor Data in 3D Latent Space', fontsize = 20)
ax.view_init(view_1, view_2)

######OLD 13 MAR 24#####
# normalize = mcolors.Normalize(vmin=labels_kmeans.min(), vmax=labels_kmeans.max())
# colormap = cm.jet
# scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
# scalarmappaple.set_array(range(n_clusters_))
# cbar = fig.colorbar(scalarmappaple, shrink = 0.6, ticks = range(n_clusters_))
################

######NEW 13 MAR 24#####
scalarmappaple = cm.ScalarMappable(cmap=cmap)
scalarmappaple.set_array(labels_kmeans)
cbar = fig.colorbar(scalarmappaple, shrink = 0.6, ticks = range(n_clusters_))
cbar.ax.set_yticklabels(range(n_clusters_), fontsize = 15)
################
plt.savefig(vae_folder + "Kmeans_all_data_{}.png".format(num_epochs), bbox_inches="tight")




#Plot Meta Data
plot_ref = {0: [0,0], 1: [0,1], 2: [1,0], 3: [1,1]}
fig, ax = plt.subplots(2, 2, figsize=(15, 15), subplot_kw=dict(projection='3d'))
fig.subplots_adjust(hspace=0.1)
for meta_type in range(4):
    if meta_type == 3:

        cmap = matplotlib.colors.ListedColormap(['green', 'red', "blue", "purple", "black", "brown", "orange"])
        cmaplist = [cmap(i) for i in range(cmap.N)]
        
        i, j = plot_ref[meta_type]
        labels_meta = meta_array[:,meta_type] #0 is odo meta
        ax[i, j].scatter(z_in[:,0], z_in[:,1], z_in[:,2], c=labels_meta, cmap=cmap)
        ax[i, j].set_title('{} Representation in 3D Latent Space'.format(meta_type_dict[meta_type]), fontsize = 15)
        ax[i, j].view_init(view_1, view_2)

        scalarmappaple = cm.ScalarMappable(cmap=cmap)
        scalarmappaple.set_array(labels_meta.astype(int))
        
        cbar = fig.colorbar(scalarmappaple, ax=ax[i, j], shrink=0.6)
        cbar.ax.set_yticklabels(list("Location {}".format(loc) for loc in location_dict.values()))
    elif meta_type == 2:
        cmap = matplotlib.colors.ListedColormap(['green', "blue", 'red', "purple", "orange"])
        cmaplist = [cmap(i) for i in range(cmap.N)]
        i, j = plot_ref[meta_type]
        labels_meta = meta_array[:,meta_type] #0 is odo meta
        ax[i, j].scatter(z_in[:,0], z_in[:,1], z_in[:,2], c=labels_meta, cmap=cmap)
        ax[i, j].set_title('{} Representation in 3D Latent Space'.format(meta_type_dict[meta_type]), fontsize = 15)
        ax[i, j].view_init(view_1, view_2)
#         normalize = mcolors.Normalize(vmin=labels_meta.min(), vmax=labels_meta.max())
#         colormap = cm.jet
#         scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
        scalarmappaple = cm.ScalarMappable(cmap=cmap)
        scalarmappaple.set_array(labels_meta.astype(int))
        
        cbar = fig.colorbar(scalarmappaple, ax=ax[i, j], shrink=0.6, ticks=[0, 1, 2 ,3, 4])
        cbar.ax.set_yticklabels(list(model_dict.keys()))
    else:
        cmap = plt.cm.jet
        cmaplist = [cmap(i) for i in range(cmap.N)]
        i, j = plot_ref[meta_type]
        labels_meta = meta_array[:,meta_type] #0 is odo meta
        ax[i, j].scatter(z_in[:,0], z_in[:,1], z_in[:,2], c=labels_meta, cmap=cmap)
        ax[i, j].set_title('{} Representation in 3D Latent Space'.format(meta_type_dict[meta_type]), fontsize = 15)
        ax[i, j].view_init(view_1, view_2)
        normalize = mcolors.Normalize(vmin=labels_meta.min(), vmax=labels_meta.max())
        colormap = cm.jet
        scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
        
        fig.colorbar(scalarmappaple, ax=ax[i, j], shrink=0.6)
plt.savefig(vae_folder + "Kmeans_meta_data_{}.png".format(num_epochs), bbox_inches="tight")