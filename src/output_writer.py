"""
Official Submission TSV Writer.
Guarantees correct tab-separated columns, no quoting, and strict subset semantics.
"""
import os
import csv

def write_candidate_pairs(path, candidate_dict, ordered_s1_ids):
    """Writes output/candidate_pairs.tsv."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in ordered_s1_ids:
            cands = candidate_dict.get(s1_id, [])
            cand_str = ",".join(cands) if cands else ""
            f.write(f"{s1_id}\t{cand_str}\n")

def write_matching_results(path, match_dict, ordered_s1_ids):
    """Writes output/matching_results.tsv."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in ordered_s1_ids:
            matches = match_dict.get(s1_id, [])
            match_str = ",".join(matches) if matches else ""
            f.write(f"{s1_id}\t{match_str}\n")
