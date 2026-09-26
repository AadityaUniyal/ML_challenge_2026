"""
Evaluation Metric: Macro F_0.5 across Source 1 entities.
"""

def compute_entity_f05(pred_set, true_set):
    if len(true_set) == 0:
        return 1.0 if len(pred_set) == 0 else 0.0
    if len(pred_set) == 0:
        return 0.0
    
    tp = len(pred_set & true_set)
    if tp == 0:
        return 0.0
    
    precision = tp / len(pred_set)
    recall = tp / len(true_set)
    
    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0
    return (1.25 * precision * recall) / denom

def evaluate_macro_f05(predictions, ground_truth):
    total_score = 0.0
    n = len(ground_truth)
    if n == 0:
        return 0.0
    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        total_score += compute_entity_f05(pred_set, true_set)
    return total_score / n
