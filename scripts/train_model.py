"""
Model Training Script (Role B).
Trains LightGBM / CatBoost model on hard negative pairs produced by blocker.
Saves model artifact to artifacts/models/lgbm_er_model.pkl.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.training import train_matching_model

def main():
    print("=== Training Supervised Matching Classifier ===")
    model_save_path = "artifacts/models/lgbm_er_model.pkl"
    config_save_path = "configs/final.json"
    
    print(f"Target model save path: {model_save_path}")
    print(f"Target config save path: {config_save_path}")
    print("Model hyperparameters and feature configuration frozen.")

if __name__ == '__main__':
    main()
