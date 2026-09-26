"""
Unified Pipeline Module integrating Role A (Blocking) and Role B (Matching).
"""
from src.normalization import normalize_record
from src.blocking import MultiChannelBlocker
from src.features import extract_pair_features
from src.inference import score_candidate_pairs, apply_decision_threshold

class EntityResolutionPipeline:
    def __init__(self, model, threshold=0.65, max_candidates=25):
        self.model = model
        self.threshold = threshold
        self.blocker = MultiChannelBlocker(max_candidates=max_candidates)
        
    def fit_target_source(self, target_records_dict):
        self.blocker.fit_targets(target_records_dict)
        
    def predict_s1_entity(self, s1_norm_rec):
        candidate_ids = self.blocker.retrieve_candidates(s1_norm_rec)
        if not candidate_ids:
            return [], []
            
        features = [
            extract_pair_features(s1_norm_rec, self.blocker.indexed_targets[cid], 10.0)
            for cid in candidate_ids
        ]
        
        probs = score_candidate_pairs(self.model, features)
        matched_ids = apply_decision_threshold(candidate_ids, probs, self.threshold)
        return candidate_ids, matched_ids
