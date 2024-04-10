# Configuration
import os
import sys
import torch
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sys import getsizeof
import json
import warnings
import random
from collections import defaultdict


from scipy import signal
from scipy import ndimage
import scipy.stats as stats
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
random.seed(10)
warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix


center = "true"#"true"#"true"
all_data_files = defaultdict(list)
# folder_attn = "/hpc/group/tarokhlab/pkk16/PhD_Research/ML_PPMx_Experiment/01-CVAE_ATTN_Combined/08VAE_ATTN-Ext_forecast_100eps_25JAN24/data/"
folder_attn = "/hpc/group/tarokhlab/pkk16/PhD_Research/ML_PPMx_Experiment/01-CVAE_ATTN_Combined/17-VAE_ATTN_30min-Confirm_20FEB24/data/"

print("num files in attn dir: ", len(os.listdir(folder_attn)))
for file in os.listdir(folder_attn):
    file_data = file.split("_")
    if file.endswith(".p") and file_data[0] == '09ga':
        ml_type, vic_num, data_point_typ, start_num, rs_worker = file_data[0][-1], file_data[2][1:], file_data[3][-1], file_data[5][1:], file_data[4][1:]
        all_data_files[(vic_num, data_point_typ, start_num, rs_worker)].append(file)
folder_cvae = "data/" + "batch_gen_cf_oos140-200_{}/".format(center)
print("num files in vae dir: ", len(os.listdir(folder_cvae)))
for file in os.listdir(folder_cvae):
    file_data = file.split("_")
    if file.endswith(".p") and file_data[0] == '09gv':
        ml_type, vic_num, data_point_typ, start_num,  rs_worker = file_data[0][-1], file_data[2][1:], file_data[3][-1], file_data[5][1:], file_data[4][1:]
        all_data_files[(vic_num, data_point_typ, start_num, rs_worker)].append(file)

common_observation_file_keys_w_error = list(all_data_files.keys())
print(len(common_observation_file_keys_w_error))
for key in common_observation_file_keys_w_error:
    if len(all_data_files[key]) < 2:
        del all_data_files[key]
print(len(all_data_files))


num_dists = 1
number_observations = len(all_data_files)
quant_nums = 1
test_fact = 0.8
FOLDS = 7
num_experiments = 50
prediction_length = 1800
quants = np.linspace(0,1,quant_nums)[1:-1] if quant_nums > 1 else np.array([0.5])
num_quants = quants.shape[0]
data_key = "data"
type_key = "type"
data_type_array = np.ones(number_observations)
all_obs_data_list, mean_gen_target_t23, true_ttf_data, true_ttf_value = [], [], [], []
obs_data_array = np.zeros((number_observations, num_dists*prediction_length*num_quants))
true_fault_data_array = np.zeros((number_observations, num_dists*prediction_length))
true_ttf_data_array = np.zeros((number_observations, 2))
for i, (obs_key, files_value) in enumerate(all_data_files.items()):
    for file in files_value:
        file_data = file.split("_")
        if file_data[0][-1] == "a":
            file_name = open(folder_attn + file, 'rb')
            output_data = pickle.load(file_name)
            data_keys = list(output_data.keys())
            data_type_array[i] = output_data[data_keys[0]][type_key]
            selected_data = output_data[data_keys[0]][data_key]
            forecasts, tss = selected_data['forecasts'], selected_data["tss"]
            obs_data_array[i, :prediction_length*num_quants] = np.hstack(np.array([forecasts[0].quantile_ts(q) for q in quants]))
            true_fault_data_array[i,:prediction_length] = np.array(tss[0][0][-prediction_length:])
            true_ttf_data_array[i,0] = tss[0][0][-prediction_length:].argmax()
        if num_dists > 1:
            if file_data[0][-1] == "v":
                file_name = open(folder_cvae + file, 'rb')
                output_data = pickle.load(file_name)
                data_keys = list(output_data.keys())
                data_type_array[i] = output_data[data_keys[0]][type_key]
                selected_data = output_data[data_keys[0]][data_key]
                forecasts, tss = selected_data['forecasts'], selected_data["tss"]
                obs_data_array[i, prediction_length*num_quants:] = np.hstack(np.array([forecasts[0].quantile_ts(q) for q in quants]))
                true_fault_data_array[i, prediction_length:] = np.array(tss[0][0][-prediction_length:])
                true_ttf_data_array[i,1] = tss[0][0][-prediction_length:].argmax()

all_X_type01 = obs_data_array
# all_y_type01 = np.abs(data_type_array - 1)
all_y_type01 = data_type_array
print(all_X_type01.shape)
print(all_y_type01.shape)

print(all_y_type01)
# assert False
all_data = np.hstack((all_X_type01, all_y_type01.reshape(all_y_type01.shape[0], 1)))
experiment_error = []
auc_array = np.zeros(num_experiments)
for k in range(num_experiments):
    np.random.shuffle(all_data)

    ttf_test_ref = round(all_data.shape[0]*test_fact)
    X_ttf_train = all_data[:,:-1][:ttf_test_ref,:]
    y_ttf_train = all_data[:,-1][:ttf_test_ref]
    X_ttf_test = all_data[:,:-1][ttf_test_ref:,:]
    y_ttf_test = all_data[:,-1][ttf_test_ref:]

    predictor = RandomForestClassifier(max_depth = 15, max_features = 5, min_samples_split = 10, n_estimators = 100)
#     predictor = svm.SVC(class_weight = "balanced",probability=True)
#         predictor = tree.DecisionTreeClassifier()
    X, y = X_ttf_train, y_ttf_train
    k_fold = KFold(n_splits=FOLDS, shuffle=True, random_state=12345)
    for i, (train_index, test_index) in enumerate(k_fold.split(X)):
        Xtrain, ytrain = X[train_index], y[train_index]
        predictor.fit(Xtrain, ytrain)

    predictor.fit(X_ttf_train, y_ttf_train)
    pred_proba = predictor.predict_proba(X_ttf_test)
    precision, recall, thresholds = precision_recall_curve(y_ttf_test, pred_proba[:,1])
    error = np.zeros((len(thresholds), 2))

    auc = roc_auc_score(y_ttf_test, pred_proba[:,1])
    auc_array[k] = auc
    for i, thresh in enumerate(thresholds):
        y_pred_new_threshold_test = (predictor.predict_proba(X_ttf_test)[:, 1] >= thresh).astype(float)
        tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_ttf_test, y_pred_new_threshold_test).ravel()
        TPR_test, TNR_test = (tp_test / (tp_test+fn_test)), (tn_test / (tn_test+fp_test))
        error[i,:] = [TNR_test, TPR_test]
    experiment_error.append(error)

f, axes = plt.subplots(1, 1, figsize=(10, 10))
y = np.linspace(0,1,100)
x = np.linspace(0,1,100)
for error_data in experiment_error:
    plt.plot(1-error_data[:,0], error_data[:,1], alpha=0.2)
plt.title("Exeriments Cond. on Vehicle State with {} Data Permutations with Avg AUC of {}".format(num_experiments, round(auc_array.mean(),3)), fontsize = 15)
plt.plot(x,y,"--", label = "Random Guess")
plt.ylabel("True Positive Rate: TP/(TP+FN)", fontsize = 15)
plt.xlabel("Fale Positive Rate: FP/(FP+TN)", fontsize = 15)
plt.legend(fontsize = 15)
plt.savefig("test_attn.png")


num_dists = 2
number_observations = len(all_data_files)
quant_nums = 1
test_fact = 0.8
FOLDS = 7
num_experiments = 50
prediction_length = 1800

quants = np.linspace(0,1,quant_nums)[1:-1] if quant_nums > 1 else np.array([0.5])
num_quants = quants.shape[0]
data_key = "data"
type_key = "type"
ttf_key = "ttf"
data_type_array = np.ones(number_observations)
all_obs_data_list, mean_gen_target_t23, true_ttf_data, true_ttf_value = [], [], [], []
obs_data_array = np.zeros((number_observations, num_dists*prediction_length*num_quants))
pred_true_data = []
for i, (obs_key, files_value) in enumerate(all_data_files.items()):
    for file in files_value:
        file_data = file.split("_")
        if file_data[0][-1] == "a":
            file_name = open(folder_attn + file, 'rb')
            output_data = pickle.load(file_name)
            data_keys = list(output_data.keys())
            data_type_array[i] = output_data[data_keys[0]][type_key]
            selected_data = output_data[data_keys[0]][data_key]
            forecasts_attn = selected_data['forecasts']
            obs_data_array[i, :prediction_length*num_quants] = np.hstack(np.array([forecasts_attn[0].quantile_ts(q) for q in quants]))

        if num_dists > 1:
            if file_data[0][-1] == "v":
                file_name = open(folder_cvae + file, 'rb')
                output_data = pickle.load(file_name)
                data_keys = list(output_data.keys())
                data_type_array[i] = output_data[data_keys[0]][type_key]
                selected_data = output_data[data_keys[0]][data_key]
                forecasts_vae, ttf  = selected_data['forecasts'], selected_data["ttf"]
                obs_data_array[i, prediction_length*num_quants:] = np.hstack(np.array([forecasts_vae[0].quantile_ts(q) for q in quants]))
                pred_true_data.append([forecasts_attn, ttf, data_type_array[i]])
                
all_X_type01 = obs_data_array
all_y_type01 = data_type_array
print(all_X_type01.shape)
print(all_y_type01.shape)

print(all_y_type01)
all_data = np.hstack((all_X_type01, all_y_type01.reshape(all_y_type01.shape[0], 1)))
experiment_error = []
auc_array = np.zeros(num_experiments)
for k in range(num_experiments):
    np.random.shuffle(all_data)

    ttf_test_ref = round(all_data.shape[0]*test_fact)
    X_ttf_train = all_data[:,:-1][:ttf_test_ref,:]
    y_ttf_train = all_data[:,-1][:ttf_test_ref]
    X_ttf_test = all_data[:,:-1][ttf_test_ref:,:]
    y_ttf_test = all_data[:,-1][ttf_test_ref:]

    predictor = RandomForestClassifier(max_depth = 15, max_features = 5, min_samples_split = 10, n_estimators = 100)
#     predictor = svm.SVC(class_weight = "balanced",probability=True)
#         predictor = tree.DecisionTreeClassifier()
    X, y = X_ttf_train, y_ttf_train
    k_fold = KFold(n_splits=FOLDS, shuffle=True, random_state=12345)
    for i, (train_index, test_index) in enumerate(k_fold.split(X)):
        Xtrain, ytrain = X[train_index], y[train_index]
        predictor.fit(Xtrain, ytrain)

    predictor.fit(X_ttf_train, y_ttf_train)
    pred_proba = predictor.predict_proba(X_ttf_test)
    precision, recall, thresholds = precision_recall_curve(y_ttf_test, pred_proba[:,1])
    error = np.zeros((len(thresholds), 2))

    auc = roc_auc_score(y_ttf_test, pred_proba[:,1])
    auc_array[k] = auc
    for i, thresh in enumerate(thresholds):
        y_pred_new_threshold_test = (predictor.predict_proba(X_ttf_test)[:, 1] >= thresh).astype(float)
        tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_ttf_test, y_pred_new_threshold_test).ravel()
        TPR_test, TNR_test = (tp_test / (tp_test+fn_test)), (tn_test / (tn_test+fp_test))
        error[i,:] = [TNR_test, TPR_test]
    experiment_error.append(error)

f, axes = plt.subplots(1, 1, figsize=(10, 10))
y = np.linspace(0,1,100)
x = np.linspace(0,1,100)
for error_data in experiment_error:
    plt.plot(1-error_data[:,0], error_data[:,1], alpha=0.2)
plt.title("Exeriments Not Cond. on Vehicle State with {} Data Permutations with Avg AUC of {}".format(num_experiments, round(auc_array.mean(),3)), fontsize = 15)
plt.plot(x,y,"--", label = "Random Guess")
plt.ylabel("True Positive Rate: TP/(TP+FN)", fontsize = 15)
plt.xlabel("Fale Positive Rate: FP/(FP+TN)", fontsize = 15)
plt.legend(fontsize = 15)
plt.savefig("test_attn_w_vae.png")
