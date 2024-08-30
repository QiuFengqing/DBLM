import torch
import torch.nn as nn
import torchsort

def Mask_Spearman_Rank_loss(allocated_weights, y_train, mask_train, k=3, regularization_strength=0.1):
    """
    Compute the Spearman rank correlation coefficient based loss in PyTorch.
    :param risk_rank: Tensor of rank list for risk.
    :param allo_rank: Tensor of rank list for allocation weight.
    :param N: Total number of items.
    :param S_kappa: Size of the set of items to be considered.
    :param eta: Regularization coefficient.
    :param Theta: Model parameters (PyTorch Tensor).
    :return: Spearman rank correlation loss.
    """

    def rank_data(data, descending=False):
        """
        Rank the data in ascending (default) or descending order.
        :param data: Tensor of data to be ranked.
        :param descending: Bool, True for descending order ranking.
        :return: Tensor of ranks.
        """
        if descending:
            data = -data
        # rank_indices = torch.argsort(data, descending=descending, dim=-1)
        # ranks = torch.argsort(rank_indices)
        ranks = torchsort.soft_rank(data, regularization_strength=regularization_strength)
        return ranks

    allocated_weights = allocated_weights.permute(0, 2, 1)

    # Apply mask
    mask = mask_train[:, :, :, 0]  # search order
    # allocated_weights[mask] = -1
    risk_weights = (y_train[:, :, :, 0] - y_train[:, :, :, 1]) ** k
    # risk_weights[mask] = torch.max(risk_weights)

    # batch
    mask = mask.reshape(-1, mask.shape[-1])
    allocated_weights = allocated_weights.reshape(-1, mask.shape[-1])
    risk_weights = risk_weights.reshape(-1, mask.shape[-1])

    allo_rank = rank_data(allocated_weights, descending=True)
    risk_rank = rank_data(risk_weights, descending=False)

    S_kappa = torch.sum(mask, dim=-1)
    N = mask.shape[-1] * torch.ones_like(S_kappa)

    squared_diff = torch.pow(risk_rank - allo_rank, 2) * mask
    squared_diff_sum = torch.sum(squared_diff, dim=1)  # Sum over items

    normalization_factor = (N - S_kappa) * ((N - S_kappa) ** 2 - 1)
    # normalization_factor_sum = torch.sum(normalization_factor, dim=1)
    spearman_loss = 6 * squared_diff_sum / (normalization_factor + 1)

    return torch.sum(spearman_loss)