from rapidfuzz import fuzz, distance

def extract_pair_features(s1_rec, cand_rec, cand_score):
    """
    Extracts 15 discriminative pairwise features between S1 and target candidate.
    """
    n1 = s1_rec['name_norm']
    n2 = cand_rec['name_norm']
    name_ratio = fuzz.ratio(n1, n2) / 100.0
    name_partial = fuzz.partial_ratio(n1, n2) / 100.0
    name_sort = fuzz.token_sort_ratio(n1, n2) / 100.0
    name_set = fuzz.token_set_ratio(n1, n2) / 100.0
    name_jw = distance.JaroWinkler.similarity(n1, n2)
    
    a1 = s1_rec['address_norm']
    a2 = cand_rec['address_norm']
    addr_ratio = fuzz.ratio(a1, a2) / 100.0
    addr_sort = fuzz.token_sort_ratio(a1, a2) / 100.0
    addr_set = fuzz.token_set_ratio(a1, a2) / 100.0
    
    nums1 = s1_rec['numbers']
    nums2 = cand_rec['numbers']
    common_nums = len(nums1 & nums2)
    diff_nums = len(nums1 ^ nums2)
    has_conflict = 1.0 if (len(nums1) > 0 and len(nums2) > 0 and common_nums == 0) else 0.0
    
    len_diff_name = abs(len(n1) - len(n2)) / (max(len(n1), len(n2)) + 1)
    len_diff_addr = abs(len(a1) - len(a2)) / (max(len(a1), len(a2)) + 1)
    
    comb1 = n1 + " " + a1
    comb2 = n2 + " " + a2
    comb_set = fuzz.token_set_ratio(comb1, comb2) / 100.0
    
    return [
        name_ratio, name_partial, name_sort, name_set, name_jw,
        addr_ratio, addr_sort, addr_set,
        float(common_nums), float(diff_nums), has_conflict,
        len_diff_name, len_diff_addr, comb_set, float(cand_score)
    ]
