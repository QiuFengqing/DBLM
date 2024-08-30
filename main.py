#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: Xinke Jiang
# @Email: XinkeJiang@stu.pku.edu.cn
# @Time: 2024/1/10 20:22
# @File: main.py

import random
import numpy as np
import warnings
import argparse
import torch
import torch.optim as optim
import torch.nn as nn

from methods.init import *
from dataloader import *
from loss_function import *
from evaluation import *

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()

# Train Args
parser.add_argument('--train_valid_test_ratio', default=[0.7,0.1,0.2], type=list, help='dataset split ratio')
parser.add_argument('--dataset_dir', default=r'data2', type=str, help='[MCM, SZ], dataset save dir')
parser.add_argument('--nhid', default=150, type=int, help='the number of hidden units')
parser.add_argument('--dropout', default=0.1, type=float, help='dropout rate')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')
parser.add_argument('--epochs', default=2000, type=int, help='the number of training epoch')
parser.add_argument('--patience', default=10, type=int, help='patience for early stopping')
parser.add_argument('--seed', default=1024, type=int, help='random seed')
parser.add_argument('--minimax', default=True, type=bool, help='whether to use minimax or not')
parser.add_argument('--train_batch_size', default=4, type=int, help='train batch size')
parser.add_argument('--valid_test_batch_size', default=2, type=int, help='test batch size')
parser.add_argument('--model', default='DBLM', type=str, help='DBLM, MLP, ECM, MSP, AGA')
parser.add_argument('--device', default='cuda:0', type=str, help='cuda, cpu')

# Model Args
parser.add_argument('--past_p', default=4, type=int, help='past time window size')
parser.add_argument('--future_f', default=4, type=int, help='future time window size')
parser.add_argument('--rank_s', default=50, type=int, help='GCN Medium Rank')
parser.add_argument('--rank_t', default=50, type=int, help='TCN Medium Rank')
parser.add_argument('--evaluation_top_k', default=10, type=int, help='TopK for HR, MRR...')
parser.add_argument('--feature_dimension', default=8, type=int, help='Feature Dimension')
parser.add_argument('--risk_k', default=2, type=int, help='Risk Power value')
parser.add_argument('--regularization_strength', default=0.5, type=int, help='Soft Rank regularization_strength')
parser.add_argument('--BLM_tau', default=3, type=int, help='Tau in BLM Model')        # 0.4
parser.add_argument('--BLM_delta', default=0.6, type=int, help='Delta in BLM Model')    # 0.6 对data1， 0.8对data2
parser.add_argument('--TCN_kernel_size', default=3, type=int, help='TCN kernel size')
parser.add_argument('--Cheb_K', default=3, type=int, help='DGCN cheb kernel')
parser.add_argument('--num_heads', default=3, type=int, help='Attention heads')
parser.add_argument('--leaky_relu_alpha', default=0.2, type=float, help='Leaky ReLU alpha')

# OTHERS
parser.add_argument('--mask_ratio', default=0, type=float, help='Mask Data')

args = parser.parse_args()

# random seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if __name__ == '__main__':
    # Read Data
    data = load_spatiotemporal_data(args.dataset_dir)

    # Split Data
    train_data, valid_data, test_data = split_train_valid_test(data, args.train_valid_test_ratio)

    # Apply Mask
    if args.mask_ratio>0:
        train_mask_ratio = torch.rand_like(train_data)
        train_data = torch.where(train_mask_ratio<args.mask_ratio, torch.zeros_like(train_mask_ratio), train_data)

    # Dataset
    train_dataset = SOS_Dataset(train_data, args.past_p, args.future_f, args.minimax, train_min_max=None)
    valid_dataset = SOS_Dataset(valid_data, args.past_p, args.future_f, args.minimax, train_min_max=train_dataset.train_min_max)
    test_dataset = SOS_Dataset(test_data, args.past_p, args.future_f, args.minimax, train_min_max=train_dataset.train_min_max)

    # Dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.valid_test_batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.valid_test_batch_size, shuffle=False, drop_last=True)

    # Model Definition
    if args.model == 'DBLM':
        model = DBLM(num_timesteps_input=args.future_f, space_dim=args.feature_dimension,
                     hidden_dim_s=args.nhid, hidden_dim_t=args.nhid, num_nodes=train_data.shape[1],
                     rank_s=args.rank_s, rank_t=args.rank_t, tau=args.BLM_tau, risk_delta=args.BLM_delta,
                     kernel_size_tcn=args.TCN_kernel_size, Cheb_K=args.Cheb_K, num_heads=args.num_heads,
                     dropout=args.dropout, leaky_relu_alpha=args.leaky_relu_alpha,
                     device=args.device)
    elif args.model == 'MLP':
        model = MLP(num_timesteps_input=args.future_f, space_dim=args.feature_dimension,
                     hidden_dim_s=args.nhid, hidden_dim_t=args.nhid, num_nodes=train_data.shape[1],
                     rank_s=args.rank_s, rank_t=args.rank_t, tau=args.BLM_tau, risk_delta=args.BLM_delta,
                     kernel_size_tcn=args.TCN_kernel_size, Cheb_K=args.Cheb_K, num_heads=args.num_heads,
                     dropout=args.dropout, leaky_relu_alpha=args.leaky_relu_alpha,
                     device=args.device)
    elif args.model == 'AGA':
        model = AGA(num_timesteps_input=args.future_f, space_dim=args.feature_dimension,
                     hidden_dim_s=args.nhid, hidden_dim_t=args.nhid, num_nodes=train_data.shape[1],
                     rank_s=args.rank_s, rank_t=args.rank_t, tau=args.BLM_tau, risk_delta=args.BLM_delta,
                     kernel_size_tcn=args.TCN_kernel_size, Cheb_K=args.Cheb_K, num_heads=args.num_heads,
                     dropout=args.dropout, leaky_relu_alpha=args.leaky_relu_alpha,
                     device=args.device)
    elif args.model == 'ECM':
        model = ECM(num_timesteps_input=args.future_f, space_dim=args.feature_dimension,
                     hidden_dim_s=args.nhid, hidden_dim_t=args.nhid, num_nodes=train_data.shape[1],
                     rank_s=args.rank_s, rank_t=args.rank_t, tau=args.BLM_tau, risk_delta=args.BLM_delta,
                     kernel_size_tcn=args.TCN_kernel_size, Cheb_K=args.Cheb_K, num_heads=args.num_heads,
                     dropout=args.dropout, leaky_relu_alpha=args.leaky_relu_alpha,
                     device=args.device)
    elif args.model == 'MSP':
        model = marketSharePrediction(indim=args.feature_dimension, dim=args.nhid, outdim=args.future_f, dev=args.device)

    # 定义优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    for epoch in range(args.epochs):
        ## Step 1, training
        """
        # Begin training, similar training procedure from STGCN
        Trains one epoch with the given data.
        :return x_train: Training inputs of shape (num_samples, num_timesteps_train, num_nodes, num_features).
        :return y_train: Training targets of shape (num_samples, num_nodes, num_nodes, 2).
        :return mask_train: Training targets mask of shape (num_samples, num_nodes, num_nodes, 2).
        :return adj_train: Training dynamic adj of shape (num_samples, num_nodes, num_nodes).
        :param batch_size: Batch size to use during training.
        """
        epoch_training_losses = []
        for x_train, y_train, mask_train, adj_train, risk_train in train_dataloader:
            x_train, y_train, mask_train, adj_train, risk_train = x_train.to(args.device), y_train.to(args.device),mask_train.to(args.device), adj_train.to(args.device), risk_train.to(args.device)

            model.train()
            optimizer.zero_grad()

            allocated_weights = model(x_train, adj_train, risk_train)

            # MSE Loss
            loss = Mask_Spearman_Rank_loss(allocated_weights, y_train, mask_train)        # DBLM, topk=10, 0.46

            # print(loss)
            loss.backward()
            optimizer.step()
            epoch_training_losses.append(loss.item())

        train_loss_epoch = sum(epoch_training_losses)/len(epoch_training_losses)

        print("Training epoch:{}, loss:{:.4f}".format(epoch, train_loss_epoch))
        # training_nll.append(sum(epoch_training_losses) / len(epoch_training_losses))
        torch.cuda.empty_cache()

        ## Step 2, validation
        with torch.no_grad():
            hr_list = []
            mrr_list = []
            ndcg_list = []
            hrs_list = []
            lrs_list = []
            blm_list = []

            valid_loss_list = []
            for x_val, y_val, mask_val, adj_val, risk_val in valid_dataloader:
            # for x_val, y_val, mask_val, adj_val, risk_val in train_dataloader:
                model.eval()
                x_val, y_val, mask_val, adj_val, risk_val = x_val.to(args.device), y_val.to(args.device),mask_val.to(args.device), adj_val.to(args.device), risk_val.to(args.device)

                # bs * T * N
                allocated_weights_val = model(x_val, adj_val, risk_val)
                # perspective_weights_val = model.P

                valid_loss = mae_loss(allocated_weights_val, y_val, mask_val)
                # valid_loss = Mask_Spearman_Rank_loss(allocated_weights_val, y_val, mask_val, k=args.risk_k, regularization_strength=args.regularization_strength) * 1000000
                valid_loss_list.append(valid_loss.item())

                # evaluation
                allocated_weights_val = allocated_weights_val.permute(0, 2, 1)
                allocated_weights_val = allocated_weights_val.reshape(-1, allocated_weights_val.shape[-1])
                risk_val = (y_val[:,:,:,0] - y_val[:,:,:,1]).reshape(-1, allocated_weights_val.shape[-1])

                # for visualization
                # np.save("perspective_weights_val_npy", perspective_weights_val.cpu().detach().numpy())
                # np.save("allocated_weights_val_npy", allocated_weights_val.cpu().detach().numpy())
                # np.save("risk_val_npy", risk_val.cpu().detach().numpy())
                # np.save("mask_val_npy", mask_val.cpu().detach().numpy())

                hr_val, mrr_val, ndcg_val, hrs_val, lrs_val, blm_val = calculate_matric(allocated_weights_val, risk_val, top_k=args.evaluation_top_k)
                hr_list.append(hr_val)
                mrr_list.append(mrr_val)
                ndcg_list.append(ndcg_val)
                hrs_list.append(hrs_val)
                lrs_list.append(lrs_val)
                blm_list.append(blm_val)

            valid_loss_result = sum(valid_loss_list)/len(valid_loss_list)
            hr_result = sum(hr_list) / len(hr_list)
            mrr_result = sum(mrr_list) / len(mrr_list)
            ndcg_result = sum(ndcg_list) / len(ndcg_list)
            hrs_result = sum(hrs_list) / len(hrs_list)
            lrs_result = sum(lrs_list) / len(lrs_list)
            blm_result = sum(blm_list) / len(blm_list)

            print("Validation loss(spearman Rank): {}, TOPK=10 HR: {}, k=20: {}, k=50: {}, MRE: {}".format(valid_loss_result, hr_result, mrr_result, ndcg_result, blm_result))

    model.eval()
    with torch.no_grad():
        hr_list = []
        mrr_list = []
        ndcg_list = []
        hrs_list = []
        lrs_list = []
        valid_loss_list = []
        for x_val, y_val, mask_val, adj_val, risk_val in test_dataloader:
            model.eval()
            x_val, y_val, mask_val, adj_val, risk_val = x_val.to(args.device), y_val.to(args.device),mask_val.to(args.device), adj_val.to(args.device), risk_val.to(args.device)

            allocated_weights_val = model(x_val, adj_val, risk_val)

            valid_loss = Mask_Spearman_Rank_loss(allocated_weights_val, y_val, mask_val)
            valid_loss_list.append(valid_loss.item())

            # evaluation
            allocated_weights_val = allocated_weights_val.permute(0, 2, 1)
            allocated_weights_val.reshape(-1, allocated_weights_val.shape[-1])
            risk_val = (y_val[:,:,:,0] - y_val[:,:,:,1]).reshape(-1, allocated_weights_val.shape[-1])

            hr_val, mrr_val, ndcg_val, hrs_val, lrs_val, blm_val = calculate_matric(allocated_weights_val, risk_val, top_k=args.evaluation_top_k)
            hr_list.append(hr_val)
            mrr_list.append(mrr_val)
            ndcg_list.append(ndcg_val)
            hrs_list.append(hrs_val)
            lrs_list.append(lrs_val)
            blm_list.append(blm_val)

        valid_loss_result = sum(valid_loss_list) / len(valid_loss_list)
        hr_result = sum(hr_list) / len(hr_list)
        mrr_result = sum(mrr_list) / len(mrr_list)
        ndcg_result = sum(ndcg_list) / len(ndcg_list)
        hrs_result = sum(hrs_list) / len(hrs_list)
        lrs_result = sum(lrs_list) / len(lrs_list)
        blm_result = sum(blm_list) / len(blm_list)

        print(
            "Test loss(spearman Rank): {}, TOPK=10 HR: {}, k=20: {}, k=50: {}, MRE: {}".format(valid_loss_result,
                                                                                                     hr_result,
                                                                                                     mrr_result,
                                                                                                     ndcg_result,
                                                                                                     blm_result))
