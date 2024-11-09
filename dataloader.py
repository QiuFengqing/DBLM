

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F


def load_spatiotemporal_data(data_path):
    order_data = np.load(data_path+'/order.npy')
    supply_data = np.load(data_path+'/supply.npy')

    data = np.concatenate((order_data[:,:,np.newaxis], supply_data[:,:,np.newaxis]), axis=-1)

    # to tensor
    data = torch.from_numpy(data)

    return data


class SOS_Dataset(Dataset):
    def __init__(self, data, p, f, perform_minimax=True, train_min_max=None):
        self.data = data    # T * N * 2: total time * total supplier *2(Order, Supply)
        self.p = p  # past time step
        self.f = f  # future time step
        self.perform_minimax = perform_minimax  # norm
        self.train_min_max = train_min_max    # train norm
        self.construct_features(data)

    def construct_features(self, data):
        T, N, _ = data.shape    # N * T * 2

        # initiation feature
        features = torch.zeros((T, N, 8))  # 7 feature

        for t in range(self.p, T):
            # f^{sv}_{it} = \mathcal{S}_{it}
            features[t, :, 0] = data[t, :, 1]   # supply

            # f^{ov}_{it} = \mathcal{O}_{it}
            features[t, :, 1] = data[t, :, 0]   # order

            # f^{sr}_{it} = \mathcal{O}_{it}-\mathcal{S}_{it}
            features[t, :, 2] = data[t, :, 0] - data[t, :, 1]

            # f^{ssv}_{it} = \mathcal{S}_{it} / \mathcal{O}_{it}
            features[t, :, 3] = (data[t, :, 1] + 1e-3) / (data[t, :, 0] + 1e-3)

            # f^{hsr}_{it} = \frac{1}{p}\sum_{t-p}^{t}(\mathcal{O}_{it}-\mathcal{S}_{it})
            features[t, :, 4] = torch.mean(data[t - self.p + 1:t + 1, :, 0].float() - data[t - self.p + 1:t + 1, :, 1].float(), dim=0)

            # f^{hssv}_{it} = \frac{1}{p} \sum_{t-p}^{t}(\mathcal{S}_{it} / \mathcal{O}_{it})
            features[t, :, 5] = torch.mean((data[t - self.p + 1:t + 1, :, 1] + 1e-3) / (data[t - self.p + 1:t + 1, :, 0] + 1e-3), dim=0)

            # f^{sc}_{it} = \max{\{\mathcal{S}_{ij}|j\in [t-p, t]\}}
            features[t, :, 6] = torch.max(data[t - self.p + 1:t + 1, :, 1], dim=0).values

            # f^{ss}_{it} = \bigl(\sum_{j=t-p}^{t}(\mathcal{S}_{ij}-f^{sc}_{it})^2/p\bigl)^{-\frac{1}{2}}
            ss_term = torch.sum((data[t - self.p + 1:t + 1, :, 1] - features[t, :, 6]) ** 2, dim=0) / self.p
            features[t, :, 7] = torch.sqrt(ss_term)

        if self.perform_minimax:
            if self.train_min_max == None:
                train_data_min = torch.min(features, dim=0)
                train_data_max = torch.max(features, dim=0)

                # Min-max
                nan_solve = train_data_max.values - train_data_min.values
                nan_solve = nan_solve + 1e-1
                features = (features - train_data_min.values) / nan_solve
                self.train_min = train_data_min.values
                self.train_max = train_data_max.values

                self.train_min_max = [self.train_min, self.train_max]

            else:
                train_min = self.train_min_max[0]
                train_max = self.train_min_max[1]

                nan_solve = train_max - train_min
                nan_solve = nan_solve + 1e-1
                # nan_solve = torch.where(nan_solve==0, 10*torch.ones_like(nan_solve), nan_solve)
                features = (features - train_min) / nan_solve

        self.features = features

    def build_dynamic_adj(self, x):

        def random_walk_normalized_laplacian(adj):
            # horizon mean
            adj = torch.mean(adj, dim=0)

            # calculate D
            degree_matrix = torch.diag(torch.sum(adj, dim=1))

            # calculate random L_rw
            inv_degree_matrix = torch.inverse(degree_matrix)
            laplacian_matrix_rw = torch.eye(adj.shape[0]) - inv_degree_matrix @ adj

            return laplacian_matrix_rw

        # compute cosine
        x_normalized = F.normalize(x, dim=-1)  # norm feature
        score_matrix = torch.matmul(x_normalized, x_normalized.transpose(-1, -2))
        adj_matrix = (score_matrix + 1) / 2

        norm_adj_matrix = random_walk_normalized_laplacian(adj_matrix)

        return norm_adj_matrix

    def __len__(self):
        return len(self.data) - self.p - self.f + 1

    def __getitem__(self, idx):
        past_idx = slice(idx, idx + self.p)
        future_idx = slice(idx + self.p, idx + self.p + self.f)

        x = torch.tensor(self.features[past_idx], dtype=torch.float32)
        y = torch.tensor(self.data[future_idx], dtype=torch.float32)
        mask = (y==0)
        adj = self.build_dynamic_adj(x)

        risk = torch.tensor(self.data[past_idx], dtype=torch.float32)

        return x, y, mask, adj, risk


def split_train_valid_test(data, train_valid_test_split=[0.7, 0.1, 0.2]):
    train_ratio, valid_ratio, test_ratio = train_valid_test_split[0], train_valid_test_split[1], train_valid_test_split[2]
    assert train_ratio + valid_ratio + test_ratio == 1.0

    data = (data - data.min()) / (data.max() - data.min())

    num_samples = len(data)
    train_size = int(num_samples * train_ratio)
    valid_size = int(num_samples * valid_ratio)
    test_size = num_samples - train_size - valid_size

    train_data = data[:train_size]
    valid_data = data[train_size:train_size+valid_size]
    test_data = data[train_size+valid_size:]

    return train_data, valid_data, test_data
