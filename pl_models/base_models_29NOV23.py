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
from sklearn.metrics import top_k_accuracy_score
from sklearn.metrics import confusion_matrix
from sparsemax import Sparsemax
import functools
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

from scipy.spatial.distance import hamming
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def dbscan_predict(model, X):
    '''dbscan doesnt natively predict cluster of unseen data use this to see'''
    nr_samples = X.shape[0]
    y_new = np.ones(shape=nr_samples, dtype=int) * -1
    for i in range(nr_samples):
        diff = model.components_ - X[i, :]  # NumPy broadcasting
        dist = np.linalg.norm(diff, axis=1)  # Euclidean distance
        shortest_dist_idx = np.argmin(dist)
        if dist[shortest_dist_idx] < model.eps:
            y_new[i] = model.labels_[model.core_sample_indices_[shortest_dist_idx]]
    return y_new


class VAE(nn.Module):
    def __init__(self, n_in, n_hid, z_dim):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc21 = nn.Linear(n_hid, z_dim)
        self.fc22 = nn.Linear(n_hid, z_dim)
        self.fc3 = nn.Linear(z_dim, n_hid)
        self.fc4 = nn.Linear(n_hid, n_in)

    def encode(self, x):
        """Encoder forward pass.
        
        Args:
            x: Input image
            
        Returns:
            mu: self.fc21(h1)
            logvar: self.fc22(h1)
        """
        
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        """Implements: z = mu + epsilon*stdev.
            
        Args: 
            mu: mean
            logvar: log of variance
        
        Return:
            z: sample from Normal(mu, var).
            
            Epsilon is sampled from standard normal distribution. 
            \epsilon \sim Normal(0, 1)
        """
        
        stdev = torch.exp(0.5*logvar)
        eps = torch.randn_like(stdev)
        return mu + eps*stdev

    def decode(self, z):
        """Decoder forward pass.
        
        Args:
            z: Batch of latent representations.
        
        Returns: 
            x_recon: Image probabilities.
        """
        
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        """Implements forward pass of VAE.
        
        Args:
            x: Batch of input images.
        
        Returns:
            x_recon: Batch of reconstructed images.
            mu: Batch of mean vectors
            logvar: Batch of log-variance vectors
        """
        
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

def train_VAE(model, device, train_loader, optimizer, epoch):
    train_loss = 0
    model.train()
    for batch_idx, (data) in enumerate(train_loader):
        data = data.view(data.size(0),-1)
        data = data.to(device)
        
        optimizer.zero_grad()
        output, mu, logvar, z = model(data)
        loss = loss_function(output, data, mu, logvar)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if batch_idx % (len(train_loader)//2) == 0:
            print('Train({})[{:.0f}%]: Loss: {:.4f}'.format(
                epoch, 100. * batch_idx / len(train_loader), train_loss/(batch_idx+1)))
    return train_loss

def test_VAE(model, device, test_loader, epoch, batch_size):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.view(data.size(0),-1)
            data = data.to(device)
            output, mu, logvar, z = model(data)
            loss = loss_function(output, data, mu, logvar)
            test_loss += loss.item() # sum up batch loss
    test_loss = (test_loss*batch_size)/len(test_loader.dataset)
    print('Test({}): Loss: {:.4f}'.format(
        epoch, test_loss))
    return test_loss

def loss_function(recon_x, x, mu, logvar):
    """Computes the loss = -ELBO = Negative Log-Likelihood + KL Divergence.
    
    Args: 
        recon_x: Decoder output.
        x: Ground truth.
        mu: Mean of Z
        logvar: Log-Variance of Z
        
        p(z) here is the standard normal distribution with mean 0 and identity covariance.
    """
    
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum') # BCE = -Negative Log-likelihood
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # KL Divergence b/w q_\phi(z|x) || p(z)
    return BCE + KLD

def train(model, device, train_loader, optimizer, epoch, num_labels = 2):
    train_loss = 0
    model.train()
    for batch_idx, data in enumerate(train_loader):
        
        features, target = data[:,:-1], data[:,-1]
        target = target.to(torch.int64) 
        #https://stackoverflow.com/questions/56513576/converting-tensor-to-one-hot-encoded-tensor-of-indices
        
        target = torch.nn.functional.one_hot(target, num_labels).to(device)#new
        
        features = features.view(features.size(0),-1).to(device)
        
        features = torch.hstack((features, target)) #new
        
#         features = features
        optimizer.zero_grad()
#         print(features, target)
#         assert False
        output, mu, logvar, z = model(features, target)
        loss = loss_function(output, features, mu, logvar)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if batch_idx % (len(train_loader)//2) == 0:
            print('Train({})[{:.0f}%]: Loss: {:.4f}'.format(
                epoch, 100. * batch_idx / len(train_loader), train_loss/(batch_idx+1)))
    return train_loss

def test(model, device, test_loader, epoch, num_labels = 2, batch_size = 8):
    model.eval()
    test_loss = 0
    with torch.no_grad():
#         for data, target in test_loader:
        for data in test_loader:
            features, target = data[:,:-1], data[:,-1]
            target = target.to(torch.int64)
            target = torch.nn.functional.one_hot(target, num_labels).to(device)#new

            features = features.view(features.size(0),-1).to(device)

            features = torch.hstack((features, target)) #new
#             features = features
            output, mu, logvar, z = model(features, target)

            loss = loss_function(output, features, mu, logvar)
            test_loss += loss.item() # sum up batch loss
    test_loss = (test_loss*batch_size)/len(test_loader.dataset)
    print('Test({}): Loss: {:.4f}'.format(
        epoch, test_loss))
    return test_loss


def make_optimizer(optimizer_name, model, **kwargs):
    if optimizer_name=='Adam':
        optimizer = optim.Adam(model.parameters(),lr=kwargs['lr'])
    elif optimizer_name=='SGD':
        optimizer = optim.SGD(model.parameters(),lr=kwargs['lr'],momentum=kwargs['momentum'], weight_decay=kwargs['weight_decay'])
    else:
        raise ValueError('Not valid optimizer name')
    return optimizer


class CVAE(nn.Module):
    def __init__(self, n_in, n_hid, z_dim, z_dim2):
        super(CVAE, self).__init__()

        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc21 = nn.Linear(n_hid, z_dim)
        self.fc22 = nn.Linear(n_hid, z_dim)
        self.fc3 = nn.Linear(z_dim2, n_hid)
        self.fc4 = nn.Linear(n_hid, n_in)

    def encode(self, x):
        """Encoder forward pass.
        
        Args:
            x: Input image
            
        Returns:
            mu: self.fc21(h1)
            logvar: self.fc22(h1)
        """
#         print(x)
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        """Implements: z = mu + epsilon*stdev.
            
        Args: 
            mu: mean
            logvar: log of variance
        
        Return:
            z: sample from Normal(mu, var).
            
            Epsilon is sampled from standard normal distribution. 
            \epsilon \sim Normal(0, 1)
        """
        
        stdev = torch.exp(0.5*logvar)
        eps = torch.randn_like(stdev)
        return mu + eps*stdev

    def decode(self, z):
        """Decoder forward pass.
        
        Args:
            z: Batch of latent representations.
        
        Returns: 
            x_recon: Image probabilities.
        """
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    
    def forward(self, x, target):
        """Implements forward pass of VAE.
        
        Args:
            x: Batch of input images.
        
        Returns:
            x_recon: Batch of reconstructed images.
            mu: Batch of mean vectors
            logvar: Batch of log-variance vectors
        """
        #print(target.shape)
        #assert False
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        z = torch.hstack((z, target))
        
        return self.decode(z), mu, logvar, z


def split_data(stock, lookback, factor):
    data_raw = stock#.to_numpy() # convert to numpy array
    data = []
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback): 
        data.append(data_raw[index: index + lookback])
    #this create 980 20x38 features to process the labels, 
    #which is the next observation after these 20 time steps.     
    
    data = np.array(data);
    test_set_size = int(np.round(factor*data.shape[0]));
    train_set_size = data.shape[0] - (test_set_size);
    
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    
    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,:]
    
    return [x_train, y_train, x_test, y_test]

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

class LSTM_sparse(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, window_size):
        super(LSTM_sparse, self).__init__()
        ####old
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        #######
        
        
        ####new
#         window_size = 599 #added
        embedding_size = self.hidden_dim #added
        num_rnn_hidden = self.num_layers #added
        context_dense_size = 38# self.hidden_dim #Just changed 3AUG
        self.rnn_hidden = self.hidden_dim 
        
        self.spatial_embed = nn.Linear(window_size,embedding_size)
        self.spatial_dense = nn.Linear(embedding_size+num_rnn_hidden,1)
        self.spatial_act = nn.Tanh()
        self.spatial_softmax = Sparsemax(dim=1) #if sparse else nn.Softmax(dim=1)  Sparsemax(dim=1)

        self.temporal_dense = nn.Linear(num_rnn_hidden, 1)
        self.temporal_act = nn.Tanh()
        self.temporal_softmax = nn.Softmax(dim=1)

        self.final_dense = nn.Linear(num_rnn_hidden, context_dense_size)
        #self.final_act = nn.ReLU()
        #######
        
        
    #def forward(self, x):
    def forward(self, x, hidden = None):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        
        # spatial attention
        #print("in", x.shape)
        #rnn_out, hidden = self.lstm(x, (h0.detach(), c0.detach()))
        rnn_out, hidden = self.lstm(x, hidden)
        #print("out", rnn_out.shape)
        rnn_out_constig = rnn_out[:,-1,:].contiguous().view(-1, self.rnn_hidden)
        #print("out2", rnn_out_constig.shape)
        spatial_embedding = self.get_spatial_embedding(x)
        betai = self.get_spatial_beta(x,rnn_out_constig, spatial_embedding=spatial_embedding)
        
        # temporal attention
        alphai = self.get_temporal_alpha(rnn_out)

        spatial_context = torch.transpose((betai/betai.mean(1).unsqueeze(1))*torch.transpose(x, 1,2), 1,2)
        #print("spatial_context", spatial_context.shape)
        
        spatial_temporal_context = (alphai/alphai.mean(1).unsqueeze(1))*spatial_context
        #spatial_temporal_out, spatial_temporal_hidden = self.rnn_net_att(spatial_temporal_context, None)
        spatial_temporal_out, spatial_temporal_hidden = self.lstm2(spatial_temporal_context, None)
        spatial_temporal_embedding = spatial_temporal_out[:,-1,:].contiguous().view(-1, self.rnn_hidden)
        #print("spatial_temporal_embedding 1", spatial_temporal_embedding.shape)
        spatial_temporal_embedding = self.final_dense(spatial_temporal_embedding)
        #print("spatial_temporal_embedding 2", spatial_temporal_embedding.shape)
        return spatial_temporal_embedding, hidden, betai

    
    
    ####new
    def get_spatial_embedding(self, x):
        x_transpose = torch.transpose(x, 1, 2)
        spatial_embedding = self.spatial_embed(x_transpose)
        
        return spatial_embedding
    
    def get_spatial_beta(self, x, rnn_out_constig, spatial_embedding=None):
        spatial_embedding = spatial_embedding if spatial_embedding is not None else self.get_spatial_embedding(x)
        #print("spatial_embedding", spatial_embedding.shape)
        spatial_concat = torch.concat([spatial_embedding, 
                                       rnn_out_constig.unsqueeze(1).repeat(1,spatial_embedding.shape[1],1)], -1)
        #print("spatial_concat",spatial_concat.shape)
        ei = self.spatial_act(self.spatial_dense(spatial_concat))
        betai = self.spatial_softmax(ei)
        
        return betai
        
    def get_temporal_alpha(self, rnn_out):
        temporal_embed = self.temporal_dense(rnn_out)
        ai = self.temporal_act(temporal_embed)
        alphai = self.temporal_softmax(ai)
        return alphai

    #######
    
##Baseline LSTM Classifier   
#https://saturncloud.io/blog/how-to-use-lstm-in-pytorch-for-classification/
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out
    
##Sparse Attention Classifier     
class LSTM_sparse_classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, num_classes):
        super(LSTM_sparse_classifier, self).__init__()
        ####old
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm3 = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, num_classes)
        #######
        
        
        ####new
        window_size = self.hidden_dim #added
        embedding_size = self.hidden_dim #added
        num_rnn_hidden = self.hidden_dim#self.num_layers #added
        context_dense_size = self.input_dim # self.hidden_dim #Just changed 3AUG
        self.rnn_hidden = self.hidden_dim 
        
        self.spatial_embed = nn.Linear(window_size,embedding_size)
        self.spatial_dense = nn.Linear(embedding_size+num_rnn_hidden,1)
        self.spatial_act = nn.Tanh()
        self.spatial_softmax = Sparsemax(dim=1) #if sparse else nn.Softmax(dim=1)  Sparsemax(dim=1)

        self.temporal_dense = nn.Linear(num_rnn_hidden, 1)
        self.temporal_act = nn.Tanh()
        self.temporal_softmax = nn.Softmax(dim=1)

        self.final_dense = nn.Linear(num_rnn_hidden, context_dense_size)
        self.sigmoid = nn.Sigmoid()
        #######
        
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        
        # spatial attention
       
        rnn_out, hidden = self.lstm(x, (h0.detach(), c0.detach()))       
        rnn_out_constig = rnn_out[:,-1,:].contiguous().view(-1, self.rnn_hidden)
        ###Added 13SEP23 to allow for variable length input
        rnn3_out, hidden = self.lstm3(x, (h0.detach(), c0.detach())) #new 13SEP23

        spatial_embedding = self.get_spatial_embedding(rnn3_out) #adjusted 13SEP23
        ###
        betai = self.get_spatial_beta(x,rnn_out_constig, spatial_embedding=spatial_embedding)
        
        # temporal attention
        alphai = self.get_temporal_alpha(rnn_out)
        spatial_context = torch.transpose((betai/betai.mean(1).unsqueeze(1))*torch.transpose(x, 1,2), 1,2)       
        spatial_temporal_context = (alphai/alphai.mean(1).unsqueeze(1))*spatial_context        
        spatial_temporal_out, spatial_temporal_hidden = self.lstm2(spatial_temporal_context, None)
        spatial_temporal_embedding = spatial_temporal_out[:,-1,:].contiguous().view(-1, self.rnn_hidden)        
        spatial_temporal_embedding = self.final_dense(spatial_temporal_embedding)        
        out = self.fc2(spatial_temporal_embedding)
        out = self.sigmoid(out)
        
        return out
    
    
    ####new
    def get_spatial_embedding(self, x):
        x_transpose = torch.transpose(x, 1, 2)
        ###Added 13SEP23 to allow for variable length input
        spatial_embedding = self.spatial_embed(x_transpose[:,:,-1])
        old_shape = tuple(spatial_embedding.shape)
        new_shape = (1,) + old_shape 
        spatial_embedding = spatial_embedding.view(new_shape)
        ###
        return spatial_embedding
    
    def get_spatial_beta(self, x, rnn_out_constig, spatial_embedding=None):
        spatial_embedding = spatial_embedding if spatial_embedding is not None else self.get_spatial_embedding(x)
#         print("spatial_embedding", spatial_embedding.shape)
#         print("rnn_out_constig", rnn_out_constig.unsqueeze(1).repeat(1,spatial_embedding.shape[1],1).shape)
        spatial_concat = torch.concat([spatial_embedding, 
                                       rnn_out_constig.unsqueeze(1).repeat(1,spatial_embedding.shape[1],1)], -1)
#         print("spatial_concat",spatial_concat.shape)
#         assert False
        ei = self.spatial_act(self.spatial_dense(spatial_concat))
        betai = self.spatial_softmax(ei)
        
        return betai
        
    def get_temporal_alpha(self, rnn_out):
        temporal_embed = self.temporal_dense(rnn_out)
        ai = self.temporal_act(temporal_embed)
        alphai = self.temporal_softmax(ai)
        return alphai

    #######
    
class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y

    def __len__(self):
        return len(self.data)
    
    
class MyDataset_CVAE():
    #from: https://stackoverflow.com/questions/45113245/how-to-get-mini-batches-in-pytorch-in-a-clean-and-efficient-way
    
    def __init__(self, x):#, y):
        super(MyDataset_CVAE, self).__init__()
        #assert x.shape[0] == y.shape[0] # assuming shape[0] = dataset size
        self.x = x
        #self.y = y


    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index]#, self.y[index]
    
def label_wise_accuracy(output, target, threshold, normed):
    target, output = target.cpu().flatten() , output.cpu().flatten() 
    pred = (output > threshold).float()
   
    ham = np.count_nonzero(target != pred)  
    CM = confusion_matrix(target, pred, normalize=normed)
    ham2 = hamming(pred, target)

    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    return TP, FP, FN, TN, ham, ham2


def split_data2(data_scaled, lookback):
    data_raw = data_scaled
    data = []
    for index in range(len(data_raw) - lookback): 
        data.append(data_raw[index: index + lookback])
    data = tensorize_list_of_tensors(data)
    x_data = data[:,:-1,:]
    y_data = data[:,-1,:]
  
    return x_data, y_data

def add_one(tens):
    old_shape = tuple(tens.shape)
    new_shape = (1,) + old_shape 
    out_tens = tens.view(new_shape)
    return out_tens

def tensorize_list_of_tensors(list_of_tensors):
    if type(list_of_tensors[0]) != np.ndarray:
        tensorize_data = np.array([tensor.cpu().detach().numpy() for tensor in list_of_tensors])
        tensorized_data = torch.from_numpy(tensorize_data).type(torch.Tensor)
    elif type(list_of_tensors[0]) == np.ndarray:
        tensorize_data = np.array(list_of_tensors)
        tensorized_data = torch.from_numpy(tensorize_data).type(torch.Tensor)
    else:
        tensorize_data = np.array([tensor.cpu().detach().numpy() for tensor in list_of_tensors])
        tensorized_data = torch.from_numpy(tensorize_data).type(torch.Tensor)
    return tensorized_data


##Baseline LSTM Classifier   
#https://saturncloud.io/blog/how-to-use-lstm-in-pytorch-for-classification/
class LSTM_classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM_classifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out
    
##Sparse Attention Classifier     
class LSTM_sparse_classifier_updated(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, num_classes):
        super(LSTM_sparse_classifier_updated, self).__init__()
        ####old
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm3 = nn.LSTM(input_dim, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, num_classes)
        #######
        
        
        ####new
        window_size = self.hidden_dim #added
        embedding_size = self.hidden_dim #added
        num_rnn_hidden = self.hidden_dim#self.num_layers #added
        context_dense_size = self.input_dim # self.hidden_dim #Just changed 3AUG
        self.rnn_hidden = self.hidden_dim 
        
        self.spatial_embed = nn.Linear(window_size, embedding_size)
        self.spatial_dense = nn.Linear(embedding_size+num_rnn_hidden,1)
        self.spatial_act = nn.Tanh()
        self.spatial_softmax = Sparsemax(dim=1) #if sparse else nn.Softmax(dim=1)  Sparsemax(dim=1)

        self.temporal_dense = nn.Linear(num_rnn_hidden, 1)
        self.temporal_act = nn.Tanh()
        self.temporal_softmax = nn.Softmax(dim=1)

        self.final_dense = nn.Linear(num_rnn_hidden, context_dense_size)
        self.sigmoid = nn.Sigmoid()
        #######
        
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        
        # spatial attention
       
        rnn_out, hidden = self.lstm(x, (h0.detach(), c0.detach()))       
        rnn_out_constig = rnn_out[:,-1,:].contiguous().view(-1, self.rnn_hidden)
        ###Added 13SEP23 to allow for variable length input
        rnn3_out, hidden = self.lstm3(x, (h0.detach(), c0.detach())) #new 13SEP23

        spatial_embedding = self.get_spatial_embedding(rnn3_out) #adjusted 13SEP23
        ###
        betai = self.get_spatial_beta(x,rnn_out_constig, spatial_embedding=spatial_embedding)
        
        # temporal attention
        alphai = self.get_temporal_alpha(rnn_out)
        spatial_context = torch.transpose((betai/betai.mean(1).unsqueeze(1))*torch.transpose(x, 1,2), 1,2)       
        spatial_temporal_context = (alphai/alphai.mean(1).unsqueeze(1))*spatial_context        
        spatial_temporal_out, spatial_temporal_hidden = self.lstm2(spatial_temporal_context, None)
        spatial_temporal_embedding = spatial_temporal_out[:,-1,:].contiguous().view(-1, self.rnn_hidden)        
        spatial_temporal_embedding = self.final_dense(spatial_temporal_embedding)        
        out = self.fc2(spatial_temporal_embedding)
        out = self.sigmoid(out)
        
        return out
    
    
    ####new
    def get_spatial_embedding(self, x):
        x_transpose = torch.transpose(x, 0, 2)

        spatial_embedding = self.spatial_embed(x_transpose)

        return spatial_embedding
    
    def get_spatial_beta(self, x, rnn_out_constig, spatial_embedding=None):
        spatial_embedding = spatial_embedding if spatial_embedding is not None else self.get_spatial_embedding(x)
        spatial_concat = torch.concat([spatial_embedding, rnn_out_constig.unsqueeze(1).repeat(1,spatial_embedding.shape[1],1)], -1)
        ei = self.spatial_act(self.spatial_dense(spatial_concat))
        betai = self.spatial_softmax(ei)
        
        return betai
        
    def get_temporal_alpha(self, rnn_out):
        temporal_embed = self.temporal_dense(rnn_out)
        ai = self.temporal_act(temporal_embed)
        alphai = self.temporal_softmax(ai)
        return alphai
