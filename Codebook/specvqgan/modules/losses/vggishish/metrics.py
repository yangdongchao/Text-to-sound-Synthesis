import logging

import numpy as np
import scipy
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

logger = logging.getLogger(f'main.{__name__}')

def metrics(targets, outputs, topk=(1, 5)):
    """
    Adapted from https://github.com/hche11/VGGSound/blob/master/utils.py

    Calculate statistics including mAP, AUC, and d-prime.
        Args:
            output: 2d tensors, (dataset_size, classes_num) - before softmax
            target: 1d tensors, (dataset_size, )
            topk: tuple
        Returns:
            metric_dict: a dict of metrics
    """
    metrics_dict = dict()

    num_cls = outputs.shape[-1]

    # accuracy@k
    _, preds = torch.topk(outputs, k=max(topk), dim=1)
    correct_for_maxtopk = preds == targets.view(-1, 1).expand_as(preds)
    for k in topk:
        metrics_dict[f'accuracy_{k}'] = float(correct_for_maxtopk[:, :k].sum() / correct_for_maxtopk.shape[0])

    # avg precision, average roc_auc, and dprime
    targets = torch.nn.functional.one_hot(targets, num_classes=num_cls)

    # ids of the predicted classes (same as softmax)
    targets_pred = torch.softmax(outputs, dim=1)

    targets = targets.numpy()
    targets_pred = targets_pred.numpy()

    # one-vs-rest
    avg_p = [average_precision_score(targets[:, c], targets_pred[:, c], average=None) for c in range(num_cls)]
    try:
        roc_aucs = [roc_auc_score(targets[:, c], targets_pred[:, c], average=None) for c in range(num_cls)]
    except ValueError:
        logger.warning('Weird... Some classes never occured in targets. Do not trust the metrics.')
        roc_aucs = np.array([0.5])
        avg_p = np.array([0])

    metrics_dict['mAP'] = np.mean(avg_p)
    metrics_dict['mROCAUC'] = np.mean(roc_aucs)
    # Percent point function (ppf) (inverse of cdf — percentiles).
    metrics_dict['dprime'] = scipy.stats.norm().ppf(metrics_dict['mROCAUC']) * np.sqrt(2)

    return metrics_dict


def metrics_tagging(targets, outputs, topk=(1, 5)):
    """
    Calculate statistics including mAP, AUC, and d-prime.
        Args:
            output: 2d tensors, (dataset_size, classes_num) - after sigmoid
            target: 2d tensors, (dataset_size, classes_num)
            topk: tuple
        Returns:
            metric_dict: a dict of metrics
    """
    metrics_dict = dict()

    # num_cls = outputs.shape[-1]

    # # accuracy@k
    # _, preds = torch.topk(outputs, k=max(topk), dim=1)
    # correct_for_maxtopk = preds == targets.view(-1, 1).expand_as(preds)
    # for k in topk:
    #     metrics_dict[f'accuracy_{k}'] = float(correct_for_maxtopk[:, :k].sum() / correct_for_maxtopk.shape[0])

    # avg precision, average roc_auc, and dprime
    # targets = torch.nn.functional.one_hot(targets, num_classes=num_cls)

    # ids of the predicted classes (same as softmax)
    # targets_pred = torch.softmax(outputs, dim=1)

    targets = targets.numpy()
    targets_pred = outputs.numpy()

    # one-vs-rest
    # avg_p = [average_precision_score(targets[:, c], targets_pred[:, c], average=None) for c in range(num_cls)]
    avg_micro = average_precision_score(targets, targets_pred, average='micro')
    # print('avg_micro ',avg_micro)
    try:
        #roc_aucs = [roc_auc_score(targets[:, c], targets_pred[:, c], average=None) for c in range(num_cls)]
        roc_aucs_micro = roc_auc_score(targets, targets_pred,average='micro')
        # print('roc_aucs ', roc_aucs_micro)
    except ValueError:
        logger.warning('Weird... Some classes never occured in targets. Do not trust the metrics.')
        roc_aucs = np.array([0.5])
        avg_p = np.array([0])

    # metrics_dict['mAP'] = np.mean(avg_p)
    # metrics_dict['mROCAUC'] = np.mean(roc_aucs)
    metrics_dict['mAP'] = avg_micro
    metrics_dict['mROCAUC'] = roc_aucs_micro
    # Percent point function (ppf) (inverse of cdf — percentiles).
    metrics_dict['dprime'] = scipy.stats.norm().ppf(metrics_dict['mROCAUC']) * np.sqrt(2)

    return metrics_dict

if __name__ == '__main__':
    targets = torch.tensor([
        [1,0,0,1],
        [1,1,0,0],
        [0,1,1,0],
        [0,0,0,1],
        [1,1,1,0],
        [1,1,1,1]
    ])
    outputs = torch.tensor([
        [0.9, 0.8,0.3,0.5],
        [0.9, 0.8,0.3,0.5],
        [0.9, 0.8,0.3,0.5],
        [0.9, 0.2,0.3,0.5],
        [0.9, 0.8,0.3,0.5],
        [0.9, 0.8,0.7,0.5],
    ]).float()
    metrics_dict = metrics_tagging(targets, outputs, topk=(1, 3))
    print(metrics_dict)
