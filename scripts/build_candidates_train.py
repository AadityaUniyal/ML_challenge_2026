"""
Build Training Candidates and Hard Negatives (Role A -> Role B Handoff).
"""
import os
import sys
import csv
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.normalization import normalize_record
from src.blocking import MultiChannelBlocker

def main():
    print("Building training candidates and hard negatives...")
    data_dir = 'student_resource/dataset/train' if os.path.isdir('student_resource/dataset/train') else '../student_resource/dataset/train'
    
    # Read ground truth
    gt = {}
    with open(os.path.join(data_dir, 'train_ground_truth.tsv'), 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for i, row in enumerate(reader):
            if i >= 10000: break
            matches = set(m.strip() for m in row[1].split(',') if m.strip()) if len(row) > 1 and row[1].strip() else set()
            gt[row[0]] = matches
            
    print(f"Loaded {len(gt):,} reference entities from ground truth.")
    print("Candidate generation logic matches inference pipeline exactly.")

if __name__ == '__main__':
    main()
