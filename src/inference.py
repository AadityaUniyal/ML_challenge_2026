"""
Inference Module (Role B).
Batched candidate scoring and F0.5 decision thresholding.
"""
import numpy as np

def score_candidate_pairs(model, feature_matrix):
    """Computes pair match probability using trained model."""
    if len(feature_matrix) == 0:
        return np.array([])
    X = np.array(feature_matrix, dtype=np.float32)
    return model.predict_proba(X)[:, 1]

def apply_decision_threshold(candidate_ids, probabilities, threshold=0.65):
    """
    Returns list of accepted matched IDs above threshold.
    Allows 0 (singleton), 1, or many matches.
    """
    accepted = []
    for cid, prob in zip(candidate_ids, probabilities):
        if prob >= threshold:
            accepted.append(cid)
    return accepted
