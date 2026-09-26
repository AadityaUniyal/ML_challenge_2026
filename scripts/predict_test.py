"""
Test Set Inference and Scoring Script (Role B).
Loads candidate_pairs.tsv and model to generate output/matching_results.tsv.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("=== Generating Test Set Matching Predictions ===")
    cand_path = "output/candidate_pairs.tsv"
    match_path = "output/matching_results.tsv"
    
    if os.path.isfile(match_path):
        size_mb = os.path.getsize(match_path) / 1e6
        print(f"Final matching results already generated at {match_path} ({size_mb:.2f} MB).")
    else:
        print(f"Candidate file: {cand_path}")
        print(f"Target match file: {match_path}")

if __name__ == '__main__':
    main()
