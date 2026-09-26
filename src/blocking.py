"""
Multi-Channel Inverted Index Candidate Generator (Blocking Engine).
Generates high-recall, bounded candidate sets per Source 1 entity across S2 and S3.
"""
from collections import defaultdict, Counter

class MultiChannelBlocker:
    def __init__(self, df_cap=2500, max_candidates=25):
        self.df_cap = df_cap
        self.max_candidates = max_candidates
        
        # Channels
        self.name_exact_idx = defaultdict(list)
        self.name_nosuffix_idx = defaultdict(list)
        self.name_compact_idx = defaultdict(list)
        self.name_word_idx = defaultdict(list)
        self.addr_word_idx = defaultdict(list)
        
        self.word_df = Counter()
        self.indexed_targets = {}

    def fit_targets(self, targets):
        """
        Indexes target records (from S2 and S3).
        Each record should be a dict produced by normalization.py.
        """
        self.indexed_targets = targets
        
        # 1. Document frequency estimation
        for tid, rec in targets.items():
            for w in set(rec['name_tokens']):
                self.word_df[('n', w)] += 1
            for w in set(rec['address_tokens']):
                self.word_df[('a', w)] += 1

        # 2. Inverted index population
        for tid, rec in targets.items():
            # Exact clean name
            if rec['name_norm']:
                self.name_exact_idx[rec['name_norm']].append(tid)
            # Exact name no suffix
            if rec['name_norm_nosuffix']:
                self.name_nosuffix_idx[rec['name_norm_nosuffix']].append(tid)
            # Compact name
            if rec['name_compact']:
                self.name_compact_idx[rec['name_compact']].append(tid)
                
            # Distinctive name tokens
            for w in rec['name_tokens']:
                if self.word_df[('n', w)] <= self.df_cap:
                    self.name_word_idx[w].append(tid)
                    
            # Distinctive address tokens
            for w in rec['address_tokens']:
                if self.word_df[('a', w)] <= self.df_cap:
                    self.addr_word_idx[w].append(tid)

    def retrieve_candidates(self, s1_rec):
        """
        Queries all blocking channels for an S1 entity and unions them with BM25-style weighting.
        Returns a list of candidate entity IDs capped at max_candidates.
        """
        cand_scores = Counter()

        # Channel 1: Exact normalized name
        if s1_rec['name_norm'] and s1_rec['name_norm'] in self.name_exact_idx:
            for tid in self.name_exact_idx[s1_rec['name_norm']]:
                cand_scores[tid] += 12

        # Channel 2: No-suffix normalized name
        if s1_rec['name_norm_nosuffix'] and s1_rec['name_norm_nosuffix'] in self.name_nosuffix_idx:
            for tid in self.name_nosuffix_idx[s1_rec['name_norm_nosuffix']]:
                cand_scores[tid] += 10

        # Channel 3: Compact name
        if s1_rec['name_compact'] and s1_rec['name_compact'] in self.name_compact_idx:
            for tid in self.name_compact_idx[s1_rec['name_compact']]:
                cand_scores[tid] += 8

        # Channel 4: Distinctive name token inverted index (IDF-weighted)
        for w in s1_rec['name_tokens']:
            df = self.word_df.get(('n', w), 0)
            if 0 < df <= self.df_cap:
                weight = 4 if df <= 50 else (2 if df <= 300 else 1)
                for tid in self.name_word_idx[w]:
                    cand_scores[tid] += weight

        # Channel 5: Address-aware alphanumeric token inverted index (IDF-weighted)
        for w in s1_rec['address_tokens']:
            df = self.word_df.get(('a', w), 0)
            if 0 < df <= self.df_cap:
                weight = 5 if df <= 30 else (3 if df <= 200 else 1)
                for tid in self.addr_word_idx[w]:
                    cand_scores[tid] += weight

        # Top-K candidate extraction
        top_cands = [tid for tid, _ in cand_scores.most_common(self.max_candidates)]
        return top_cands
