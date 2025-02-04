from __future__ import annotations
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
import csv
import random
import numpy as np
import torch.utils.data
import os
import logging
from datetime import datetime
import math
import time
import copy
import sys
import h5py
import argparse
import configparser
import matplotlib.pyplot as plt
import pandas as pd
import scipy.linalg as la
from sklearn.cluster import KMeans
from typing import Union, Optional, Tuple

# Dataset related functions
def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                           shuffle=shuffle, drop_last=drop_last)
    return dataloader

class MinMaxNorm01(object):
    """scale data to range [0, 1]"""
    def __init__(self):
        pass

    def fit(self, x):
        self.min = x.min()
        self.max = x.max()

    def transform(self, x):
        x = 1.0 * (x - self.min) / (self.max - self.min)
        return x

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        x = x * (self.max - self.min) + self.min
        return x

# Model related classes
@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    vocab_size: int
    seq_in: int
    seq_out: int
    d_state: int = 128
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 3
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight

class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embedding = nn.Linear(args.vocab_size, args.d_model)
        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)
        self.in_proj_r = nn.Linear(args.d_model, args.d_inner, bias=args.bias)
        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        self.norm_f = RMSNorm(args.d_model)
        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)
        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

    def forward(self, x):
        # Implementation of forward pass
        x = self.norm_f(x)
        x_and_res = self.in_proj(x)
        (x, res) = x_and_res.chunk(2, dim=-1)
        x = self.ssm(x)
        x = x * F.silu(res)
        return self.out_proj(x)

    def ssm(self, x):
        # Implementation of SSM
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, :-(self.args.d_conv-1)]
        x = rearrange(x, 'b d l -> b l d')
        return x

class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.mixer = MambaBlock(args)
        self.norm = nn.LayerNorm(args.d_model)

    def forward(self, x):
        return x + self.mixer(self.norm(x))

class SAMBA(nn.Module):
    def __init__(self, ModelArgs, hidden, inp, out, embed, cheb_k):
        super().__init__()
        self.args = ModelArgs
        self.mam1 = MambaBlock(ModelArgs)
        self.cheb_k = cheb_k
        self.gamma = nn.Parameter(torch.tensor(1.))
        self.adj = nn.Parameter(torch.randn(ModelArgs.vocab_size, embed), requires_grad=True)
        self.embed_w = nn.Parameter(torch.randn(embed, embed), requires_grad=True)
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed, cheb_k, inp, out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed, out))
        self.proj = nn.Linear(ModelArgs.vocab_size, 1)
        self.proj_seq = nn.Linear(ModelArgs.seq_in, 1)

    def forward(self, input_ids):
        xx = self.mam1(input_ids)
        ADJ = self.gaussian_kernel_graph(self.adj, xx, gamma=self.gamma)
        I = torch.eye(input_ids.size(2)).cuda()
        support_set = [I, ADJ]
        
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * ADJ, support_set[-1]) - support_set[-2])
        
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', self.adj, self.weights_pool)
        bias = torch.matmul(self.adj, self.bias_pool)
        x_g = torch.einsum("knm,bmc->bknc", supports, xx.permute(0,2,1))
        x_g = x_g.permute(0, 2, 1, 3)
        out = torch.einsum('bnki,nkio->bno', x_g, weights) + bias
        
        return self.proj(out.permute(0,2,1))

    def gaussian_kernel_graph(self, E_A, x, gamma=1.0):
        x_mean = torch.mean(x, dim=0)
        x_time = torch.mm(x_mean.permute(1,0), x_mean)
        N = E_A.size(0)
        E_A_expanded = E_A.unsqueeze(0).expand(N, N, -1)
        E_A_T_expanded = E_A.unsqueeze(1).expand(N, N, -1)
        distance_matrix = torch.sum((E_A_expanded - E_A_T_expanded)**2, dim=2)
        A = torch.exp(-gamma * distance_matrix)
        dr = nn.Dropout(0.35)
        A = F.softmax(A, dim=1)
        return dr(A)

class SchrodingerClustering:
    """
    Implements quantum-inspired clustering using Schrödinger PCA approach.
    This combines quantum mechanics principles with traditional clustering methods.
    """
    def __init__(self, n_clusters: int = 8, n_components: int = 2, hbar: float = 1.0, mass: float = 1.0):
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.hbar = hbar  # Planck's constant
        self.mass = mass  # Particle mass parameter
        self.eigenvalues = None
        self.eigenvectors = None
        self.clusters = None
        self.kmeans = KMeans(n_clusters=n_clusters)
        
    def _compute_hamiltonian(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute the Hamiltonian operator for the quantum system.
        H = -ℏ²/2m ∇² + V(x)
        """
        n_samples = X.shape[0]
        # Compute pairwise distances
        distances = torch.cdist(X, X)
        # Kinetic energy term (Laplacian)
        laplacian = -self.hbar**2 / (2 * self.mass) * torch.exp(-distances**2)
        # Potential energy term (using distance matrix as potential)
        potential = torch.diag(torch.sum(distances, dim=1))
        return laplacian + potential
    
    def fit(self, X: torch.Tensor) -> 'SchrodingerClustering':
        """
        Fit the Schrödinger clustering model to the data.
        
        Args:
            X: Input data tensor of shape (n_samples, n_features)
        """
        # Convert to CPU for eigendecomposition
        X_cpu = X.cpu() if X.is_cuda else X
        
        # Compute Hamiltonian
        H = self._compute_hamiltonian(X_cpu)
        
        # Solve the time-independent Schrödinger equation (eigenvalue problem)
        eigenvalues, eigenvectors = torch.linalg.eigh(H)
        
        # Sort by eigenvalues
        idx = torch.argsort(eigenvalues)
        self.eigenvalues = eigenvalues[idx]
        self.eigenvectors = eigenvectors[:, idx]
        
        # Use the first n_components eigenvectors for dimensionality reduction
        reduced_data = self.eigenvectors[:, :self.n_components]
        
        # Perform k-means clustering in the reduced space
        self.clusters = self.kmeans.fit_predict(reduced_data)
        
        return self
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict clusters for new data points.
        
        Args:
            X: Input data tensor of shape (n_samples, n_features)
            
        Returns:
            Cluster assignments
        """
        if self.eigenvectors is None:
            raise RuntimeError("Model must be fitted before making predictions")
            
        H = self._compute_hamiltonian(X)
        reduced_data = torch.matmul(H, self.eigenvectors[:, :self.n_components])
        return torch.tensor(self.kmeans.predict(reduced_data))
    
    def fit_predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Fit the model and predict clusters in one step.
        
        Args:
            X: Input data tensor of shape (n_samples, n_features)
            
        Returns:
            Cluster assignments
        """
        self.fit(X)
        return torch.tensor(self.clusters)

class SAMBAWithClustering(SAMBA):
    """
    Extended SAMBA model with Schrödinger clustering capabilities
    """
    def __init__(self, ModelArgs, hidden, inp, out, embed, cheb_k, n_clusters=8):
        super().__init__(ModelArgs, hidden, inp, out, embed, cheb_k)
        self.clustering = SchrodingerClustering(n_clusters=n_clusters)
        
    def forward_with_clustering(self, input_ids):
        # Get the base SAMBA embeddings
        embeddings = self.mam1(input_ids)
        
        # Perform clustering on the embeddings
        clusters = self.clustering.fit_predict(embeddings.view(-1, embeddings.size(-1)))
        clusters = clusters.view(embeddings.size(0), -1)
        
        # Regular forward pass
        output = self.forward(input_ids)
        
        return output, clusters

def cluster_analysis(model: SAMBAWithClustering, data_loader: torch.utils.data.DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform clustering analysis on the data using the SAMBA model.
    
    Args:
        model: SAMBAWithClustering model
        data_loader: DataLoader containing the data
        
    Returns:
        Tuple of (predictions, cluster_assignments)
    """
    model.eval()
    predictions = []
    clusters = []
    
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(data_loader):
            output, batch_clusters = model.forward_with_clustering(data)
            predictions.append(output)
            clusters.append(batch_clusters)
    
    return torch.cat(predictions, dim=0), torch.cat(clusters, dim=0)

# Training related functions
def get_logger(root, name=None, debug=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s: %(message)s', "%Y-%m-%d %H:%M")
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if not debug:
        logfile = os.path.join(root, 'run.log')
        file_handler = logging.FileHandler(logfile, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

def init_seed(seed):
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def MAE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true-pred))

def MSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean((pred - true) ** 2)

def RMSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.mean((pred - true) ** 2))

class Trainer(object):
    def __init__(self, model, loss, optimizer, train_loader, val_loader, test_loader,
                 args, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        if val_loader != None:
            self.val_per_epoch = len(val_loader)
        self.best_path = os.path.join(self.args.get('log_dir'), 'best_model.pth')
        self.loss_figure_path = os.path.join(self.args.get('log_dir'), 'loss.png')
        #log
        if os.path.isdir(args.get('log_dir')) == False and not args.get('debug'):
            os.makedirs(args.get('log_dir'), exist_ok=True)
        self.logger = get_logger(args.get('log_dir'), name='train', debug=args.get('debug'))
        self.logger.info('Experiment log path in: {}'.format(args.get('log_dir')))
        #if not args.debug:
        #self.logger.info("Argument: %r", args)
        for arg, value in sorted(vars(args).items()):
            self.logger.info("Argument %s: %r", arg, value)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            output = self.model(data)
            if self.args.get('real_value'):
                output = self.scaler.inverse_transform(output)
            loss = self.loss(output, target)
            loss.backward()
            if self.args.get('grad_norm'):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.get('max_grad_norm'))
            self.optimizer.step()
            total_loss += loss.item()

            #log information
            if batch_idx % self.args.get('log_step') == 0:
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
                    epoch, batch_idx, self.train_per_epoch, loss.item()))
        train_epoch_loss = total_loss/self.train_per_epoch
        self.logger.info('**********Train Epoch {}: averaged Loss: {:.6f}, tf_ratio: {:.6f}'.format(epoch, train_epoch_loss, 0))

        #learning rate decay
        if self.args.get('lr_decay'):
            self.lr_scheduler.step()
        return train_epoch_loss

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                output = self.model(data)
                if self.args.get('real_value'):
                    output = self.scaler.inverse_transform(output)
                loss = self.loss(output, target)
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        val_loss = total_val_loss / len(val_dataloader)
        self.logger.info('**********Val Epoch {}: averaged Loss: {:.6f}'.format(epoch, val_loss))
        return val_loss

    def train(self):
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        for epoch in range(1, self.args.get('epochs') + 1):
            train_epoch_loss = self.train_epoch(epoch)
            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            val_epoch_loss = self.val_epoch(epoch, val_dataloader)

            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            if train_epoch_loss > 1e6:
                self.logger.info("Gradient explosion detected. Ending...")
                break
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            # early stop
            if self.args.get('early_stop'):
                if not_improved_count == self.args.get('early_stop_patience'):
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                "Training stops.".format(self.args.get('early_stop_patience')))
                    break
            # save the best state
            if best_state == True:
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())

        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time/60), best_loss))

        #save the best model to file
        if not self.args.get('debug'):
            torch.save(best_model, self.best_path)
            self.logger.info("Saving current best model to " + self.best_path)

        #test
        self.model.load_state_dict(best_model)
        #self.val_epoch(self.args.epochs, self.test_loader)
        self.test(self.model, self.args, self.test_loader, self.logger)

    def test(self, model, args, data_loader, logger, path=None):
        if path != None:
            check_point = torch.load(path)
            model.load_state_dict(check_point)
            model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data
                label = target
                output = model(data)
                y_true.append(label)
                y_pred.append(output)
        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)
        
        return y_pred, y_true
