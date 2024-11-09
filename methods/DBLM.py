#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Time: 2024/1/10 19:42
# @File: DBLM.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import sys
import math
from scipy.stats import nbinom
from torch.nn.utils import weight_norm


class D_GCN(nn.Module):
    """
    Neural network block that applies a diffusion graph convolution to sampled location
    """

    def __init__(self, in_channels, out_channels, orders, activation='relu'):
        """
        :param in_channels: Number of time step.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param order: The diffusion steps.
        """
        super(D_GCN, self).__init__()
        self.orders = orders
        self.activation = activation
        self.num_matrices = self.orders + 1
        self.Theta1 = nn.Parameter(torch.FloatTensor(in_channels * self.num_matrices,
                                                     out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)
        stdv1 = 1. / math.sqrt(self.bias.shape[0])
        self.bias.data.uniform_(-stdv1, stdv1)

    def _concat(self, x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def forward(self, X, A_q):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps)
        :A_q: The forward random walk matrix (num_nodes, num_nodes)
        :return: Output data of shape (batch_size, num_nodes, num_features)
        """
        batch_size = X.shape[0]  # batch_size
        num_node = X.shape[2]
        input_size = X.shape[1]  # time_length
        feature_size = X.shape[3]
        
        supports = []
        A_q = (A_q.unsqueeze(1)).repeat([1, input_size, 1, 1]).reshape(batch_size*input_size, num_node, num_node)       # assign time * batchsize
        supports.append(A_q)
        
        # bs * t * n * F
        x0 = X
        x0 = torch.reshape(x0, shape=[batch_size*input_size, num_node, feature_size])     # (bs, n, TF)
        x = torch.unsqueeze(x0, 0)
        for support in supports:
            x1 = torch.bmm(support, x0)
            x = self._concat(x, x1)     # bs, N, F
            for k in range(2, self.orders + 1):
                x2 = 2 * torch.bmm(support, x1) - x0
                x = self._concat(x, x2)
                x1, x0 = x2, x1
        
        # x: nm * bs * n * F

        # x: nm, bs*t, N, F
        x = x.permute(1, 2, 3, 0)       # bs*t, n, F, nm
        x = x.reshape(batch_size*input_size, num_node, feature_size*self.num_matrices)
        x = x.reshape(batch_size, input_size, num_node, feature_size*self.num_matrices)
        
        x = torch.matmul(x, self.Theta1)  # (batch_size * self._num_nodes, output_size)
        x += self.bias
        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'selu':
            x = F.selu(x)

        return x

class B_TCN(nn.Module):
    """
    Neural network block that applies a bidirectional temporal convolution to each node of
    a graph, considering additional feature dimensions.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu', device='cpu'):
        """
        :param in_channels: Number of input features (F).
        :param out_channels: Desired number of output features.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(B_TCN, self).__init__()
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.activation = activation
        self.device = device
        # Adjusted to handle the feature dimension F as in_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

        self.conv1b = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2b = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3b = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_timesteps, num_nodes, num_features)
        :return: Output data of shape (batch_size, num_timesteps, num_nodes, num_features)
        """
        batch_size, seq_len, num_nodes, num_features = X.shape
        # bs * T * N * F
        Xf = X.permute(0, 3, 2, 1)      # bs, F, N, T
        
        # Inverse the direction of time for backward convolution
        inv_idx = torch.arange(Xf.size(2) - 1, -1, -1).long().to(self.device)
        Xb = Xf.index_select(2, inv_idx)

        # Forward and backward convolutions
        tempf = self.conv1(Xf) * torch.sigmoid(self.conv2(Xf))
        outf = tempf + self.conv3(Xf)
        outf = outf.permute(0, 3, 2, 1)  # bs, seq_len - self.kernel_size + 1, self.out_channels, num_features

        tempb = self.conv1b(Xb) * torch.sigmoid(self.conv2b(Xb))
        outb = tempb + self.conv3b(Xb)

        # bs, F , N, T
        outb = outb.permute(0, 3, 2, 1) # bs, seq_len - self.kernel_size + 1, self.out_channels, num_features

        # Reconstruct the original sequence length
        rec = torch.zeros_like(outf).to(self.device)

        outf = torch.cat((outf, rec), dim=1)
        outb = torch.cat((outb, rec), dim=1)

        # Inverse index for backward output
        inv_idx = torch.arange(outb.size(1) - 1, -1, -1).long().to(self.device)
        outb = outb.index_select(1, inv_idx)

        # Combine forward and backward outputs
        if self.activation == 'relu':
            out = F.relu(outf) + F.relu(outb)
        elif self.activation == 'sigmoid':
            out = F.sigmoid(outf) + F.sigmoid(outb)
        else:
            out = None

        return out


class FusionAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_nodes, num_heads=3, dropout=0.1, leaky_relu_alpha=0.2):
        super(FusionAttention, self).__init__()
        self.num_heads = num_heads
        
        # Learnable matrices
        self.W_B_attn = nn.Linear(2*input_size, hidden_size*self.num_heads)

        # Fusion weights
        self.W_fu = nn.Linear(2*input_size, num_nodes, bias=False)
        self.a = nn.Linear(2*hidden_size*self.num_heads, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(leaky_relu_alpha)
        
        self.dropout = dropout


    def forward(self, H, E, adj):
        # H and E are the input features from nodes i and j
        torch.cuda.empty_cache()

        # bs * T * N * 2F
        ST_Feature = torch.cat([H, E], dim=-1)
        bs, T, N, feature_f = ST_Feature.shape[0], ST_Feature.shape[1], ST_Feature.shape[2], ST_Feature.shape[3]
        # A_hat is the adjacency matrix for the graph at time t
        adj = (adj.unsqueeze(1)).repeat([1, T, 1, 1]).reshape(bs * T, N, N)
        
        h = self.W_B_attn(ST_Feature)  # [batch_size, T, N, out_features]
        h = h.view(bs * T, N, -1)  # Reshape for attention computation
        torch.cuda.empty_cache()

        # Calculate attention coefficients: bs * T, N, F  => bs*T, N, N, 2F
        h = h.unsqueeze(-2)
        h = h.repeat(1, 1, N, 1)
        a_input = torch.cat([h, h], dim=1).view(bs * T, N, N, -1)
        torch.cuda.empty_cache()
        e = self.leakyrelu(self.a(a_input)).squeeze(-1)

        # Masking + Softmax
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout)

        # Get P
        P_tmp = self.W_fu(ST_Feature).reshape(bs * T, N, -1)
        P = torch.tanh(torch.bmm(attention, torch.bmm(adj, P_tmp)))
        return P.reshape(bs, T, N, N)


class DBLM(nn.Module):
    def __init__(self, num_timesteps_input=4, space_dim=8, hidden_dim_s=60, num_nodes=1, hidden_dim_t=64, rank_s=20,
                 rank_t=4, risk_k=2, tau=1, risk_delta=0.6, past_p=4, future_f=4, kernel_size_tcn=3, Cheb_K=3, num_heads=3,
                 dropout=0.1, leaky_relu_alpha=0.2,
                 device='cpu'):
        """
        DBLM Model Implementation
        """
        super(DBLM, self).__init__()
        self.device = device
        # TCN
        self.TCN1 = B_TCN(in_channels=space_dim, out_channels=hidden_dim_t, kernel_size=kernel_size_tcn, device=device).to(device=device)
        self.TCN2 = B_TCN(hidden_dim_t, rank_t, kernel_size=kernel_size_tcn, device=device).to(device=device)
        self.TCN3 = B_TCN(rank_t, hidden_dim_t, kernel_size=kernel_size_tcn, device=device).to(device=device)

        # BCN
        self.SCN1 = D_GCN(in_channels=space_dim, out_channels=hidden_dim_s, orders=Cheb_K).to(device=device)
        self.SCN2 = D_GCN(hidden_dim_s, rank_s, Cheb_K, activation='linear').to(device=device)
        self.SCN3 = D_GCN(rank_s, hidden_dim_s, Cheb_K).to(device=device)

        # Fusion
        self.FusionAttention = FusionAttention(hidden_dim_t, hidden_dim_s, num_nodes,
                                               num_heads=num_heads, dropout=dropout, leaky_relu_alpha=leaky_relu_alpha).to(device=device)
        
        # Omega
        self.W_om = nn.Linear(num_nodes, num_nodes, bias=False).to(device=device)

        # Predict
        self.W_out = nn.Linear(past_p, future_f).to(device=device)
        
        self.past_p = past_p
        self.future_f = future_f
        self.num_nodes = num_nodes

        self.tau = tau
        self.k = risk_k
        self.risk_delta = risk_delta

        # Ablation Study
        self.MLP_linear = nn.Linear(space_dim, hidden_dim_s).to(device=device)
    
    def BLM_calculator(self, P, Y):
        # Get Omega
        # bs, T, N
        bs, T, N = P.shape[0], P.shape[1], P.shape[2]
        Sigma_diag = ((Y[:, :, :, 0] - Y[:, :, :, 1]) ** self.k).to(self.device)
        
        Sigma_diag = Sigma_diag.reshape(bs*T, N, 1)
        # restore Sigma_diag to Sigma
        Sigma_tmp = (torch.ones(1, N).unsqueeze(0).repeat([bs*T, 1, 1])).to(self.device)
        Sigma = torch.bmm(Sigma_diag, Sigma_tmp) + 1e-3         # keep positive-definite: refer to appendix
        Sigma = torch.diagonal(Sigma, dim1=-2, dim2=-1)
        Sigma = torch.diag_embed(Sigma)
        
        P = P.reshape(bs*T, N, N)
        
        Pi = torch.ones_like(Sigma[:, :, 0]).to(self.device)    # bs*T, n
        Pi = torch.softmax(Pi, dim=-1)
        gauss_noise = torch.randn(Pi.size()).to(self.device)
        Q = Pi + gauss_noise    #  regularization calibration

        Omega_tmp = torch.bmm(torch.bmm(P, Sigma), P.transpose(-1, -2))
        Omega_tmp_2 = torch.clip(torch.sigmoid(self.W_om(Omega_tmp)), 0, 1-1e-4)     # avoid 0
        # bs*T, N, N
        Omega = torch.diagonal(Omega_tmp_2, dim1=-2, dim2=-1)      # diag_embd is to keep semi-define
        Omega = torch.diag_embed(Omega)
        
        # Compute
        inverse_tmp = torch.bmm(torch.bmm(P, (self.tau * Sigma)), P.transpose(-1, -2)) + Omega
        inverse_term = torch.inverse(inverse_tmp)

        # Compute hat_mu
        hat_mu_tmp = torch.bmm(Sigma, P.transpose(-1, -2))
        hat_mu_tmp_1 = torch.bmm(hat_mu_tmp, inverse_term)
        hat_mu_tmp_2 = torch.bmm(P, Pi.unsqueeze(-1))
        hat_mu_tmp_3 = torch.bmm(hat_mu_tmp_1, (Q.unsqueeze(-1) - hat_mu_tmp_2))
        hat_mu = Pi + self.tau * hat_mu_tmp_3.squeeze(-1)

        # Compute hat_Sigma
        hat_Sigma_tmp = torch.bmm(Sigma, P.transpose(-1, -2))
        hat_Sigma_tmp_1 = torch.bmm(hat_Sigma_tmp, inverse_term)
        hat_Sigma_tmp_2 = torch.bmm(hat_Sigma_tmp_1, P)
        hat_Sigma_tmp_3 = torch.bmm(hat_Sigma_tmp_2, self.tau * Sigma)
        
        hat_Sigma = (1 + self.tau) * Sigma - self.tau * hat_Sigma_tmp_3

        return hat_mu, hat_Sigma

    def BLM_solver(self, hat_mu, hat_Sigma):
        
        inv_term = torch.inverse(self.risk_delta * hat_Sigma)

        # Compute W_BLM
        W_BLM = torch.bmm(inv_term, hat_mu.unsqueeze(-1))

        return W_BLM.squeeze(-1)

    def forward(self, X, A, history_Y):
        """
        :param X: Input data of shape (batch_size, num_timesteps, num_nodes, num_features)
        :A_hat: The Laplacian matrix (num_nodes, num_nodes)
        :return: Reconstructed X of shape (batch_size, num_timesteps, num_nodes)
        """

        X_T = X
        # stack three layers
        X_t1 = self.TCN1(X_T)
        X_t2 = self.TCN2(X_t1)  # num_time, rank
        self.temporal_factors = X_t2
        X_t3 = self.TCN3(X_t2)

        _b, _t, _n, _f = X_t3.shape

        # stack three layers
        X_s1 = self.SCN1(X, A)
        X_s2 = self.SCN2(X_s1, A)  # num_nodes, rank
        self.space_factors = X_s2
        X_s3 = self.SCN3(X_s2, A)

        _b, _t, _n, _f = X_s3.shape

        # Fusion => P
        P = self.FusionAttention(X_t3, X_s3, A)
        self.P = P

        # => Mu, Sigma
        hat_mu, hat_Sigma = self.BLM_calculator(P, history_Y)

        # Solve BLM
        W_BLM_history = self.BLM_solver(hat_mu, hat_Sigma)
        W_BLM_history = W_BLM_history.reshape(-1, self.past_p, self.num_nodes).transpose(1, 2)

        # Predict
        allocated_weights = torch.softmax(self.W_out(W_BLM_history), dim=1)        # norm

        return allocated_weights.relu()
