# models/metrics.py

import numpy as np
from sklearn.metrics import average_precision_score

def compute_ap(pred, target):
    try:
        return average_precision_score(target, pred)
    except:
        return 0.0

def compute_tta(pred, target, threshold=0.5):
    """
    Time-to-Accident (TTA) = first frame where pred > threshold before true accident starts
    """
    if target.sum() == 0:
        return 0.0  # non-accident video

    acc_start = target.nonzero()[0].item()
    for t in range(acc_start):
        if pred[t] > threshold:
            return acc_start - t

    return 0.0

def compute_metrics(preds, labels, threshold=0.5):
    ap = compute_ap(preds, labels)
    mtta = compute_tta(preds, labels, threshold)
    return {
        'AP': ap,
        'mTTA': mtta
    }
