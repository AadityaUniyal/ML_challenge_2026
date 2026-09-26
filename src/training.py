"""
Training Module for Matching Classifier (Role B).
Trains LightGBM classifier on hard-negative and positive candidate pairs.
"""
import os
import pickle
import numpy as np
import lightgbm as lgb

def train_matching_model(X_train, y_train, model_params=None, save_path=None):
    if model_params is None:
        model_params = {
            'n_estimators': 400,
            'learning_rate': 0.06,
            'num_leaves': 45,
            'max_depth': 8,
            'min_child_samples': 50,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
    
    clf = lgb.LGBMClassifier(**model_params)
    clf.fit(X_train, y_train)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(clf, f)
            
    return clf

def load_matching_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)
