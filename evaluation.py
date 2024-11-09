
import torch
import torchsort

def calculate_hr(predictions, targets, top_k=1, threshold=0.5):
    """
    Calculate Hit Rate (HR) considering only the Top K predictions
    """
    top_k = min((targets>0).sum().item(), top_k)
    _, top_indices = torch.topk(predictions, top_k, largest=False)     # min weight
    _, top_indices_target = torch.topk(targets, top_k)     # max risk
    binary_predictions = torch.zeros_like(predictions)
    target_predictions = torch.zeros_like(predictions)
    binary_predictions[top_indices] = 1
    target_predictions[top_indices_target] = 1

    hits = (binary_predictions * target_predictions).sum().item()      # prop
    hits = hits / (top_k+1)     # avoid smaller than top_k

    return hits

def loss_metric(predictions, targets, top_k, ascending=True):
    risk_sum = torch.sum(predictions * targets)
    return risk_sum.item()

def calculate_matric(predictions_list, targets_list, top_k=10):
    # predictions_list = torch.softmax(predictions_list, dim=-1)
    hr_list = []
    mrr_list = []
    ndcg_list = []
    hrs_list = []
    lrs_list = []
    blm_list = []
    for i in range(len(predictions_list)):
        predictions = predictions_list[i]
        targets = targets_list[i]

        hr = calculate_hr(predictions, targets, top_k=top_k)
        mrr = calculate_hr(predictions, targets, top_k=20)
        ndcg = calculate_hr(predictions, targets, top_k=50)
        blm_risk = loss_metric(predictions, targets, top_k=top_k)

        hr_list.append(hr)
        mrr_list.append(mrr)
        ndcg_list.append(ndcg)
        blm_list.append(blm_risk)

    hr_result = sum(hr_list) / len(hr_list)
    mrr_result = sum(mrr_list) / len(mrr_list)
    ndcg_result = sum(ndcg_list) / len(ndcg_list)
    hrs_result = sum(hrs_list) / len(hrs_list)
    lrs_result = sum(lrs_list) / len(lrs_list)
    blm_result = sum(blm_list) / len(blm_list)

    return hr_result, mrr_result, ndcg_result, hrs_result, lrs_result, blm_result
