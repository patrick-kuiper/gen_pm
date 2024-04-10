# Configuration
'''
Trains a sinlge model for each vehicle, specific to a fault type
-Saves the loss history
-no data type: just starts at time = 0
-run for 100 epochs
-64 batch size based on paper
'''
import os
import sys
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


from gluonts.model.deepvar import DeepVAREstimator
from gluonts.mx.distribution import MultivariateGaussianOutput
from gluonts.mx.trainer import Trainer


from gluonts.mx.trainer import Trainer
from gluonts.model import deepar
from gluonts.evaluation import make_evaluation_predictions, Evaluator
import mxnet as mx
from gluonts.mx.trainer.callback import TrainingHistory
from sklearn.cluster import KMeans


########Define Inputs#################
fault_type_dict = {"combined": 39, "brakes": 40, "engine": 41, "transmission": 42} 
fault_select = "combined"
vic_num = int(os.environ['SLURM_ARRAY_TASK_ID'])
eps = 5
targets_dim = 4
#base vehicle data 
data_length = 129600 #36 hours of data
prediction_len = 1800
####*****UPDATE START EVAL TO START*********##############
start = 0
prediction_start = start +  data_length #prediction start
target_num_start = 39
feature_nums = [i for i in range(1,target_num_start)]

## Define Other Folders (depend on Random List)####
#Full True Vehicle Data Folder
folder_true_data = "/hpc/home/pkk16/tarokhlab/pkk16/PhD_Research/PPMx_Data_processing/01-Updated_Data_Analysis_24MAY23/04-Clean_Label/seperate_fault_data/"
file_true_data = "adjusted_fault_features_combined_one_label_{}.p".format(vic_num)
#Folders To Save New Data
model_folder = "models_train_ft_{}/".format(fault_select)
vic_save_num = "vic_model_vic{}_st{}_ft{}/".format(vic_num, start, fault_select)

###########Make directories###############
if not os.path.isdir(model_folder): 
    os.mkdir(model_folder)
out_model_folder = model_folder + vic_save_num
if not os.path.isdir(out_model_folder): 
    os.mkdir(out_model_folder)
    
#############Pull True Vehicle Data########### 
file_name_true_data = open(folder_true_data + file_true_data, 'rb')
all_vehicle_data = pickle.load(file_name_true_data)
file_name_true_data.close()

#############Get data to be put into trainer###########
type_data, output_data = all_vehicle_data

features = output_data[:,1:-targets_dim]
targets = output_data[:,fault_type_dict[fault_select]].reshape(-1, 1)
times = output_data[:,0].reshape(-1, 1)

scaler_features, scaler_target = StandardScaler(), StandardScaler()
scaler_features.fit(features)
scaler_target.fit(targets)
data_features_scaled = scaler_features.transform(features)
data_target_scaled = scaler_target.transform(targets)

#########Combine data ######################
all_data_scaled = np.hstack((times.reshape(times.shape[0],1), data_features_scaled, 
                             data_target_scaled.reshape(data_target_scaled.shape[0],1)))
df = pd.DataFrame(all_data_scaled)
df = df.set_index(list(df)[0])

num_feat_real = len(feature_nums)
history = TrainingHistory()
trainer = Trainer(
  ctx=mx.cpu(),
  epochs = eps,
  callbacks=[history],
#   batch_size = 2,
#   num_batches_per_epoch = 1,
      batch_size = 64,
      num_batches_per_epoch = 50,
  learning_rate = 2e-3,
  patience = 5,
  minimum_learning_rate = 1e-5,
  clip_gradient = 10.0,
  weight_decay = 1e-4,
  hybridize = False,
)

estimator = deepar.DeepAREstimator(freq="h",
                            prediction_length = prediction_len,
                            context_length = prediction_len,
                            trainer=trainer,
                            use_feat_dynamic_real=True,
                            )

training_data = ListDataset(
    [{"start": df.index[0], "target": df[target_num_start][df.index[0]:df.index[data_length]],
      'feat_dynamic_real': [df[i][df.index[start]:df.index[data_length]] for i in feature_nums]
      }],
    freq="h"
)

predictor = estimator.train(training_data = training_data)
predictor.serialize(Path(out_model_folder))

history_info = history.loss_history
hist_file_name = "loss_hist_vic{}_st{}_ft{}.p".format(vic_num, start, fault_select)
with open(out_model_folder + hist_file_name, 'wb') as f: #each file will have one vehicles base data in it
    pickle.dump(history_info, f)