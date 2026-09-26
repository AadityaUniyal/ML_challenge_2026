"""
Pairwise Feature Engineering Module (Role B).
Computes 20 discriminative signals across:
- Name exact, edit distance, token-level Jaccard, token sort/set, Jaro-Winkler
- Address exact, token-level Jaccard, token sort/set, length differences
- Alphanumeric building & plot number overlap, symmetric difference, conflict flags
- Cross-feature interactions (strong name + strong address, exact name + conflict address)
- Open-set country agreement flag
"""
from rapidfuzz import fuzz, distance

def compute_jaccard(tokens1, tokens2):
    s1, s2 = set(tokens1), set(tokens2)
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / float(len(s1 | s2))

def compute_overlap_ratio(tokens1, tokens2):
    s1, s2 = set(tokens1), set(tokens2)
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / float(min(len(s1), len(s2)))

def extract_pair_features(s1_rec, cand_rec, cand_score=0.0):
    """
    Extracts 20 pairwise numerical features between S1 reference record and target candidate.
    """
    n1 = s1_rec['name_norm']
    n2 = cand_rec['name_norm']
    a1 = s1_rec['address_norm']
    a2 = cand_rec['address_norm']
    
    # 1-5. Name similarity signals
    name_ratio = fuzz.ratio(n1, n2) / 100.0
    name_partial = fuzz.partial_ratio(n1, n2) / 100.0
    name_sort = fuzz.token_sort_ratio(n1, n2) / 100.0
    name_set = fuzz.token_set_ratio(n1, n2) / 100.0
    name_jw = distance.JaroWinkler.similarity(n1, n2)
    
    # 6-7. Name token overlap & Jaccard
    name_jaccard = compute_jaccard(s1_rec['name_tokens'], cand_rec['name_tokens'])
    name_overlap_ratio = compute_overlap_ratio(s1_rec['name_tokens'], cand_rec['name_tokens'])
    
    # 8-10. Address similarity signals
    addr_ratio = fuzz.ratio(a1, a2) / 100.0
    addr_sort = fuzz.token_sort_ratio(a1, a2) / 100.0
    addr_set = fuzz.token_set_ratio(a1, a2) / 100.0
    addr_jaccard = compute_jaccard(s1_rec['address_tokens'], cand_rec['address_tokens'])
    
    # 11-13. Number overlap and conflict signals
    nums1 = s1_rec['numbers']
    nums2 = cand_rec['numbers']
    common_nums = float(len(nums1 & nums2))
    diff_nums = float(len(nums1 ^ nums2))
    has_conflict = 1.0 if (len(nums1) > 0 and len(nums2) > 0 and common_nums == 0) else 0.0
    
    # 14-15. Length differences
    len_diff_name = abs(len(n1) - len(n2)) / (max(len(n1), len(n2)) + 1)
    len_diff_addr = abs(len(a1) - len(a2)) / (max(len(a1), len(a2)) + 1)
    
    # 16. Combined text token set ratio
    comb1 = n1 + " " + a1
    comb2 = n2 + " " + a2
    comb_set = fuzz.token_set_ratio(comb1, comb2) / 100.0
    
    # 17. Country agreement (open-set comparison)
    country_match = 1.0 if s1_rec.get('country', '').lower() == cand_rec.get('country', '').lower() else 0.0
    
    # 18. Interaction: Strong name + Strong address
    strong_both = 1.0 if (name_jw >= 0.85 and addr_set >= 0.75) else 0.0
    
    # 19. Interaction: Exact name + Conflicting address (Chain/Branch penalty)
    exact_name_conflict = 1.0 if (name_ratio >= 0.90 and has_conflict == 1.0) else 0.0
    
    # 20. Blocker score prior
    prior_score = float(cand_score)
    
    return [
        name_ratio, name_partial, name_sort, name_set, name_jw,
        name_jaccard, name_overlap_ratio,
        addr_ratio, addr_sort, addr_set, addr_jaccard,
        common_nums, diff_nums, has_conflict,
        len_diff_name, len_diff_addr, comb_set,
        country_match, strong_both, exact_name_conflict,
        prior_score
    ]

FEATURE_NAMES = [
    'name_ratio', 'name_partial', 'name_sort', 'name_set', 'name_jw',
    'name_jaccard', 'name_overlap_ratio',
    'addr_ratio', 'addr_sort', 'addr_set', 'addr_jaccard',
    'common_nums', 'diff_nums', 'has_conflict',
    'len_diff_name', 'len_diff_addr', 'comb_set',
    'country_match', 'strong_both', 'exact_name_conflict',
    'prior_score'
]
