# Configuration
'''
Variables:
-Data type
-random sample number
Built to run on random slices of missions, where these have been selected according to the appropriate distirbution of tests
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

from gluonts.dataset.util import to_pandas
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import ListDataset
from sklearn.preprocessing import StandardScaler
from gluonts.mx.distribution.neg_binomial import NegativeBinomialOutput
from sklearn.preprocessing import MinMaxScaler
from pl_models.base_models_29NOV23 import CVAE

from gluonts.mx.trainer import Trainer
from gluonts.model import deepar
from gluonts.evaluation import make_evaluation_predictions, Evaluator
import mxnet as mx
from sklearn.cluster import KMeans
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


########Define Inputs#################
#experiment data
gen_type = 'a'
start_vic = int(os.environ['SLURM_ARRAY_TASK_ID'])
end_vic = start_vic + 1

fault_type_dict = {"combined": 39, "brakes": 40, "engine": 41, "transmission": 42} 
fault_select = "combined"
selected_fault_col_num = fault_type_dict[fault_select]

rs_worker_start = 10
rs_worker_end = 15
targets_dim = 4
deepar_train_start = 0
eps = 100
#base vehicle data 
attn_train_len = 10800
data_length = 129600 #36 hours of data
prediction_len = 1800
target_num = 39
feature_nums = [i for i in range(1,target_num)]
#ATTN Data
num_epochs_attn = 200
#Data for loading the CVAE data
num_epochs_cvae = 1000
z_dim = 3
input_dim = 38
hidden_dim = 3
num_layers = 3
output_dim = 38
lookback = 1200

folder_true_data = "vehicle_data/"
randomlist_folder = "random_lists/"
vae_folder = "vae_data/"
data_folder = "data/"
deepar_model_folder = "models_train_ft_{}/".format(fault_select)

for vic_name in range(start_vic, end_vic):
    print("#############################")
    print("on vehicle: ", vic_name)
    start_data_dict = {}
    #############Pull Random List Data########### 
    #Random List Folder
    file_true_data = "adjusted_fault_features_combined_one_label_{}.p".format(vic_name)
    file_rand_list = "uniform_random_start_list_type_v{}_tl{}_pl{}.p".format(vic_name, data_length, prediction_len)
    
    file_name_rand_list = open(randomlist_folder + file_rand_list, 'rb')
    randomlist = pickle.load(file_name_rand_list)
    file_name_rand_list.close()
    try:
        file_name_rand_list = open(randomlist_folder + file_rand_list, 'rb')
        randomlist = pickle.load(file_name_rand_list)
        file_name_rand_list.close()
    except:
        print("no random list")
        continue
    ######START DATA#######################
    T = deepar_train_start +  data_length
    vic_save_num = "vic_model_vic{}_st{}_ft{}/".format(vic_name, deepar_train_start, fault_select)
    
    for rs_worker in range(rs_worker_start, rs_worker_end):
        attn_start = randomlist[rs_worker]
        pred_start = attn_start + attn_train_len
        print("start: ", deepar_train_start)
        print("T ", T)
        print("attn_start", attn_start)
        if attn_start < T:
            print("pass this data point")
            continue
        ### Define Other Folders (depend on Random List)####
        #Full True Vehicle Data Folder
        file_true_data = "adjusted_fault_features_combined_one_label_{}.p".format(vic_name)
        #Folders To Save New Data

        
        ####*****UPDATE START EVALL TO START*********##############
        file_gen_data = "07g{}_Generated_data_v{}_s{}_lk{}_ep{}_pl{}.p".format(gen_type, vic_name, attn_start, lookback, num_epochs_attn, prediction_len)
        ###########Make directories###############
        out_model_folder = deepar_model_folder + vic_save_num
        if not os.path.isdir(data_folder): 
            os.mkdir(data_folder)
        #############Pull True Vehicle Data########### 
        file_name_true_data = open(folder_true_data + file_true_data, 'rb')
        all_vehicle_data = pickle.load(file_name_true_data)
        file_name_true_data.close()
        #############Pull Generated Vehicle Data###########
        try:
            file_name_gen_data = open(data_folder + file_gen_data, 'rb')
            gen_data = pickle.load(file_name_gen_data)
            file_name_gen_data.close()
            print(type(gen_data), gen_data.shape)
        except:
            print("not in generated dataset")
            continue
        #############Get data to be put into trainer###########
        vic_serial, output_data = all_vehicle_data
        
        features = output_data[:,1:-targets_dim ]
        target = output_data[:, selected_fault_col_num].reshape(-1, 1)
        times = output_data[:,0]
        scaler_features = StandardScaler()
        scaler_target = StandardScaler()
        scaler_features.fit(features)
        scaler_target.fit(target)
        data_features_scaled = scaler_features.transform(features)
        data_target_scaled = scaler_target.transform(target)
        print("tragets: ", data_target_scaled.shape)
        print("times: ", times.shape)
        print("features: ", data_features_scaled.shape)
        #########Combine data ######################

        all_data_scaled = np.hstack((times.reshape(times.shape[0],1), data_features_scaled, 
                                     data_target_scaled.reshape(data_target_scaled.shape[0],1)))
########Updated 31JAN23: Added dummy target data to ensure no data snooping######
        dummy_target_data = np.zeros_like(data_target_scaled.reshape(data_target_scaled.shape[0],1)[pred_start : pred_start + prediction_len])
        gen_data_scaled = np.hstack((times.reshape(times.shape[0],1)[pred_start : pred_start + prediction_len], 
                                     gen_data, dummy_target_data))
########################################################################################
        train_test_data_scaled = np.vstack((all_data_scaled[deepar_train_start : data_length], gen_data_scaled))
        
        print("final shape: ", train_test_data_scaled.shape)
        df = pd.DataFrame(train_test_data_scaled)
        df = df.set_index(list(df)[0])
        test_data = ListDataset(
                [
                    {"start": df.index[df.index == df.index[data_length]][0], "target": df[target_num][df.index[0]:df.index[data_length+prediction_len-1]],
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
        sample_type_test = target[pred_start : pred_start + prediction_len].sum()
        if sample_type_test > 0:
            type_of_sample = 1
        else:
            type_of_sample = 0
        print("type of sample: ", type_of_sample)
        print(target[pred_start : pred_start + prediction_len].sum())
        plt.plot(target[pred_start : pred_start + prediction_len])
        plt.savefig("type_test_tp{}.png".format(type_of_sample))
        
        model_dict_data = {"forecasts": forecasts, "tss": tss}
        start_data_dict[(attn_start, vic_name, type_of_sample)] = {"data": model_dict_data, "type": type_of_sample}
        print("Saving Data")
        file_name = data_folder + '09g{}_output_v{}_ty{}_w{}_s{}_dl{}_pl{}_ep{}.p'.format(gen_type, vic_name, type_of_sample, rs_worker, deepar_train_start, data_length, prediction_len, eps)
        with open(file_name, 'wb') as f: #each file will have one vehicles base data in it
            pickle.dump(start_data_dict, f)