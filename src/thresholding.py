"""
Threshold Optimization Module for Macro F_0.5.
"""
import numpy as np
from src.evaluation import evaluate_macro_f05

def optimize_threshold(val_probabilities_by_s1, val_ground_truth, search_range=np.arange(0.30, 0.85, 0.05)):
    best_thresh = 0.65
    best_score = 0.0

    for thresh in search_range:
        preds = {}
        for s1_id, p_list in val_probabilities_by_s1.items():
            preds[s1_id] = {tid for tid, prob in p_list if prob >= thresh}
        
        score = evaluate_macro_f05(preds, val_ground_truth)
        if score > best_score:
            best_score = score
            best_thresh = thresh

    return float(best_thresh), float(best_score)
