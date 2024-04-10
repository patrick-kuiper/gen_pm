# Configuration
'''
This script returns a generated fault trace
INPUT:
1. data type {0,1}
2. K-means center for analysis {0,...,4}
3. Vehicle under analysis 
4. Sample for model
5. Random samples to be generated over
OUTPUT:
1. 
'''
import os
import sys
from pts import Trainer
import torch
from gluonts.evaluation.backtest import make_evaluation_predictions
import pickle
import pandas as pd
import numpy as np
import random
from pathlib import Path
from gluonts.model.predictor import Predictor

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib import style

from sys import getsizeof
import json
import warnings
warnings.filterwarnings("ignore")
from joblib import dump, load

from gluonts.dataset.util import to_pandas
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import ListDataset
from sklearn.preprocessing import StandardScaler
# from gluonts.mx.distribution.neg_binomial import NegativeBinomialOutput
from sklearn.preprocessing import MinMaxScaler

from pl_models.base_models_29NOV23 import CVAE, VAE

from gluonts.mx.trainer import Trainer
from gluonts.model import deepar
from gluonts.evaluation import make_evaluation_predictions, Evaluator
import mxnet as mx
from sklearn.cluster import KMeans
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

########Define Inputs#################
#experiment data
center = "true" #"true" #number 0-4 or "true"
num_epochs = 100

vic_start = int(os.environ['SLURM_ARRAY_TASK_ID'])
vic_end = vic_start + 1
gen_type = 'v'

rs_worker_start = 10
rs_worker_end = 50
deepar_train_start = 0
targets_dim = 4

target_num = 39 #define to pull last column for fault - this works because the correct type of fault has been ulled
eps = 100
num_reruns = 1
#base vehicle data 
data_length = 129600
prediction_len = 1800
attn_train_len = 10800
feature_nums = [i for i in range(1,39)]
#ATTN Data
num_epochs_attn = 200
#Data for loading the CVAE data
num_epochs_cvae = 100
z_dim = 3
input_dim = 38
hidden_dim = 3
num_layers = 3
output_dim = 38
lookback = 1200
num_attributes = 39
num_features = 38
num_steps_pred = 1800

num_labels = 2
lr = 0.001
n_in = (num_features * num_steps_pred)  #new
n_hid = 400  
z_dim = 3 
z_dim2 = z_dim
fault_type_dict = {"combined": 39, "brakes": 40, "engine": 41, "transmission": 42} 
fault_select = "combined"
######################################################

vae_folder = "vae_data/"
folder_true_data = "vehicle_data/"

test_type_folder = "batch_gen_cf_oos140-200_{}/".format(center) #chaved from: 
deepar_model_folder = "models_train_ft_{}/".format(fault_select)

data = "data/"
randomlist_folder = "random_lists/"

#Load VAE Model
vae_load = VAE(n_in, n_hid, z_dim).to(device)
###BELOW ALL ARE LOADED FROM PREVIOUS FOLDER
vae_load.load_state_dict(torch.load(vae_folder + 'VAE_model_eps{}_zdim{}.pt'.format(num_epochs, z_dim), map_location=torch.device(device)))
#Load Kmeans Model
with open(vae_folder + 'Kmeans_model_eps{}_zdim{}.p'.format(num_epochs, z_dim), 'rb') as f:
    kmeans_model = pickle.load(f)

    
out_vae_data_folder = data + test_type_folder

if not os.path.isdir(data): 
    os.mkdir(data)
if not os.path.isdir(out_vae_data_folder): 
    os.mkdir(out_vae_data_folder)

string_errors = []
test_feature_data = {}
for vic_name in range(vic_start,vic_end):
    ################Load and Scale Data######################
    print("on vic: ", vic_name)
    start_data_dict = {}
    file_true_data = "adjusted_fault_features_combined_one_label_{}.p".format(vic_name)
    file_rand_list = "uniform_random_start_list_type_v{}_tl{}_pl{}.p".format(vic_name, data_length, prediction_len)
    try:
        file_name_true_data = open(folder_true_data + file_true_data, 'rb')
        all_vehicle_data = pickle.load(file_name_true_data)
        file_name_true_data.close()
    except:
        print("no base vehicle data")
        continue  
    #############Pull Generated Vehicle Data###########
    try:
        file_name_rand_list = open(randomlist_folder + file_rand_list, 'rb')
        randomlist = pickle.load(file_name_rand_list)
        file_name_rand_list.close()
    except:
        print("no random list")
        continue 
    
    selected_fault_col_num = fault_type_dict[fault_select]
    vic_serial, output_data = all_vehicle_data
    features = output_data[:, 1:-targets_dim]
    target_all_data = output_data[:, selected_fault_col_num].reshape(-1, 1)
    times = output_data[:, 0]
    scaler_features = MinMaxScaler(feature_range=(0, 1))#StandardScaler()
    scaler_target = StandardScaler()
    scaler_features.fit(features)
    scaler_target.fit(target_all_data)
    data_features_scaled = scaler_features.transform(features)
    data_target_scaled = scaler_target.transform(target_all_data)
    all_data_scaled = np.hstack((times.reshape(times.shape[0],1), data_features_scaled, 
                                     data_target_scaled.reshape(data_target_scaled.shape[0],1)))
    
    #############Pull Generated Vehicle Data###########
    T = deepar_train_start +  data_length
    vic_save_num = "vic_model_vic{}_st{}_ft{}/".format(vic_name, deepar_train_start, fault_select)
    out_model_folder = deepar_model_folder + vic_save_num
    
    for rs_worker in range(rs_worker_start, rs_worker_end):
        attn_start = randomlist[rs_worker]
        print("start: ", deepar_train_start)
        print("T ", T)
        print("attn_start", attn_start)
        if attn_start < T:
            print("pass this data point")
            continue
        pred_start = attn_start + attn_train_len
        selected_data = all_data_scaled[pred_start : pred_start + prediction_len, 1 : target_num].reshape(1, -1)
     
    #############Generate Data from Trained Model###########
        if center == "true":
            #pull data from true value and get associated latent space data
            decode_z, _, _, out_z = vae_load(torch.tensor(selected_data.astype(float)).to(torch.float32))
            out_z = out_z.detach().numpy()
            try:
                pred_center = kmeans_model.predict(out_z)
            except:
                print("pass this point: vae output nan")
                continue
        else:
            #############################################################################################
            #Get Cluster centers for entire data set to randomly sample from, for the given data type    
            ##UPDATED 11JAN14
            #Load Base VAE Data
            file_name = open(vae_folder + 'test_feature_data.p', 'rb')
            gen_data_vae = pickle.load(file_name)
            file_name.close()
            gen_data_keys = list(gen_data_vae.keys())
            data_list = []
            for i, gen_key in enumerate(gen_data_keys):
                if gen_data_vae[gen_key].shape[0] != (num_attributes * num_steps_pred):
                    continue
                else:
                    data_list.append(gen_data_vae[gen_key])
            data_array =  np.array(data_list).reshape((len(data_list), num_steps_pred, num_attributes))

            #Rescal Base VAE Data
            data, target = data_array[:, :, :-1].reshape((data_array.shape[0], num_features * num_steps_pred)), data_array[:, :, -1]
            index_w_fault = np.where(target.max(1) == 1)[0]
            data_w_target = np.hstack((data, target.max(1).reshape(data.shape[0],-1))).astype("float32") #stach sensor data with fault data for scaling
            #Scale Data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(data_w_target[:,:-1])
            data_w_target[:,:-1] = scaler.transform(data_w_target[:,:-1])

            #Decode using VAE Model
            vae_input_data = torch.tensor(data_w_target[:,:-1])
            decode_z, _, _, out_z = vae_load(vae_input_data)
            #Get attributes of trained KMEANS Model
            labels_kmeans = kmeans_model.labels_
            noise_mask = [labels_kmeans != -1]
            keep_z = out_z[noise_mask]
            keep_z_labels = labels_kmeans[noise_mask]
            n_clusters_ = len(set(labels_kmeans)) - (1 if -1 in labels_kmeans else 0)
            #Randomly select data point from defined cluster
            vae_latent_sample_points = np.hstack((keep_z.detach().numpy(), keep_z_labels.reshape(-1, 1))).reshape(-1, z_dim + 1)#Select the random point
            C_i = [np.where(vae_latent_sample_points[:,z_dim] == i)[0].tolist() for i in range(n_clusters_)]
            #Input the random point and draw sample in z_dim
            out_z_with_cluster = vae_latent_sample_points[np.random.choice(C_i[center])]
            out_z, pred_center = out_z_with_cluster[:-1].astype(np.float32), out_z_with_cluster[-1].astype(np.float32)

        #########Determine K-Means Center################
        print(out_z)
        print(pred_center)
        print(type(out_z))
        print(vae_load.decode(torch.tensor(out_z)).reshape(prediction_len, -1).shape)
        gen_data_scaled = vae_load.decode(torch.tensor(out_z)).reshape(prediction_len, -1)
        ###NEW RESCALING
        print("before unscaling: ", gen_data_scaled)
        UN_gen_data_scaled = scaler_features.inverse_transform(gen_data_scaled.detach().numpy())
        print("after unscaling: ", UN_gen_data_scaled)
        RE_scaler_features = StandardScaler()
        RE_scaler_features.fit(UN_gen_data_scaled)
        gen_data_scaled = RE_scaler_features.transform(UN_gen_data_scaled)
        print("Rescaling Normal: ", gen_data_scaled)
        #get true data in 
########Updated 31JAN23: Added dummy target data to ensure no data snooping######
        dummy_target_data = np.zeros_like(data_target_scaled.reshape(data_target_scaled.shape[0],1)[pred_start : pred_start + prediction_len])
        gen_data_scaled = np.hstack((times.reshape(times.shape[0],1)[pred_start : pred_start + prediction_len], 
                                     gen_data_scaled, dummy_target_data))
########################################################################################
        train_test_data_scaled = np.vstack((all_data_scaled[deepar_train_start : data_length], gen_data_scaled))
#         train_test_data_scaled = np.vstack((all_data_scaled[attn_start : pred_start], gen_data_scaled))
        print(train_test_data_scaled.shape)
        #########Train DeepAR Model################
        df = pd.DataFrame(train_test_data_scaled)
        df = df.set_index(list(df)[0])

        test_data = ListDataset(
                [
                    {"start": df.index[df.index == df.index[data_length]][0], "target": df[target_num][df.index[0]:df.index[data_length + prediction_len - 1]],
                     'feat_dynamic_real': [df[i][df.index[0]:df.index[data_length + prediction_len - 1]] for i in feature_nums]
                     }
                ],
                freq="h"
            )
        try:
            predictor_deserialized = Predictor.deserialize(Path(out_model_folder))
        except:
            print("no DeepAR model available")
            continue
        forecast_it, ts_it = make_evaluation_predictions(test_data, predictor = predictor_deserialized, 
                                                         num_samples = 100)
        forecasts = list(forecast_it)
        tss = list(ts_it)
        
        #Determine type of sample from true target data during the prediction period
        sample_type_test = target_all_data[pred_start : pred_start + prediction_len].sum()
        ttf = target_all_data[pred_start : pred_start + prediction_len].argmax()
        if sample_type_test > 0:
            type_of_sample = 1
        else:
            type_of_sample = 0
        print("type of sample: ", type_of_sample)
        print(target_all_data[pred_start : pred_start + prediction_len].sum())
        plt.plot(target_all_data[pred_start : pred_start + prediction_len])
        plt.title("TTF {}".format(ttf))
        plt.savefig("type_test_tp{}.png".format(type_of_sample))
        
        model_dict_data = {"forecasts": forecasts, "tss": tss, "ttf": ttf}
        start_data_dict[(attn_start, vic_name, type_of_sample)] = {"data": model_dict_data, "type": type_of_sample, "center": pred_center}
        print("Saving Data")
        file_name = out_vae_data_folder + '09g{}_output_v{}_ty{}_w{}_s{}_dl{}_pl{}_ep{}.p'.format(gen_type, vic_name, type_of_sample, rs_worker, deepar_train_start, data_length, prediction_len, eps)
        with open(file_name, 'wb') as f: #each file will have one vehicles base data in it
            pickle.dump(start_data_dict, f)
    
