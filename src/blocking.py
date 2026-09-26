"""
Advanced Multi-Channel Blocking Engine (Role A).
Implements the full 7-channel retrieval strategy:
1. Exact normalized name
2. Suffix-normalized name
3. Compact name
4. Character n-gram shingle retrieval (soft character similarity)
5. Distinctive token inverted index (IDF-weighted token retrieval)
6. Token overlap retrieval
7. Address-aware alphanumeric anchor retrieval
Treats country as an open-set label.
"""
from collections import defaultdict, Counter

def get_char_ngrams(text, n=3):
    """Generates character n-grams from normalized text."""
    if len(text) < n:
        return [text] if text else []
    return [text[i:i+n] for i in range(len(text) - n + 1)]

class MultiChannelBlocker:
    def __init__(self, df_cap=2500, max_candidates=25):
        self.df_cap = df_cap
        self.max_candidates = max_candidates
        
        # Inverted index channels
        self.name_exact_idx = defaultdict(list)
        self.name_nosuffix_idx = defaultdict(list)
        self.name_compact_idx = defaultdict(list)
        self.name_token_idx = defaultdict(list)
        self.name_char_ngram_idx = defaultdict(list)
        self.addr_token_idx = defaultdict(list)
        
        self.token_df = Counter()
        self.char_ngram_df = Counter()
        self.indexed_targets = {}

    def fit_targets(self, targets):
        """
        Indexes target records (S2 and S3) for a specific country or open set.
        """
        self.indexed_targets = targets
        
        # 1. Estimate document frequencies for tokens and character n-grams
        for tid, rec in targets.items():
            for w in set(rec['name_tokens']):
                self.token_df[('n', w)] += 1
            for w in set(rec['address_tokens']):
                self.token_df[('a', w)] += 1
                
            # Distinctive character 3-grams from name (for typos/OCR errors)
            fcn = rec.get('name_norm_nosuffix', '')
            if fcn:
                ngrams = set(get_char_ngrams(fcn, 3))
                for ng in ngrams:
                    self.char_ngram_df[ng] += 1

        # 2. Build multi-channel inverted indexes
        for tid, rec in targets.items():
            # Channel 1: Exact clean name
            if rec['name_norm']:
                self.name_exact_idx[rec['name_norm']].append(tid)
            # Channel 2: No-suffix name
            if rec['name_norm_nosuffix']:
                self.name_nosuffix_idx[rec['name_norm_nosuffix']].append(tid)
            # Channel 3: Compact name
            if rec['name_compact']:
                self.name_compact_idx[rec['name_compact']].append(tid)
                
            # Channel 4: Distinctive name tokens
            for w in rec['name_tokens']:
                if self.token_df[('n', w)] <= self.df_cap:
                    self.name_token_idx[w].append(tid)
                    
            # Channel 5: Character n-grams (cap DF to avoid ultra-common n-grams)
            fcn = rec.get('name_norm_nosuffix', '')
            if fcn:
                for ng in set(get_char_ngrams(fcn, 3)):
                    if 5 <= self.char_ngram_df[ng] <= 1500:
                        self.name_char_ngram_idx[ng].append(tid)
                        
            # Channel 6 & 7: Address tokens & alphanumeric building anchors
            for w in rec['address_tokens']:
                if self.token_df[('a', w)] <= self.df_cap:
                    self.addr_token_idx[w].append(tid)

    def retrieve_candidates(self, s1_rec):
        """
        Retrieves top candidate IDs for an S1 entity by unioning all 7 channels with IDF weights.
        """
        cand_scores = Counter()

        # Pass 1: Exact normalized name (High precision anchor)
        if s1_rec['name_norm'] and s1_rec['name_norm'] in self.name_exact_idx:
            for tid in self.name_exact_idx[s1_rec['name_norm']]:
                cand_scores[tid] += 12

        # Pass 2: No-suffix normalized name
        if s1_rec['name_norm_nosuffix'] and s1_rec['name_norm_nosuffix'] in self.name_nosuffix_idx:
            for tid in self.name_nosuffix_idx[s1_rec['name_norm_nosuffix']]:
                cand_scores[tid] += 10

        # Pass 3: Compact name
        if s1_rec['name_compact'] and s1_rec['name_compact'] in self.name_compact_idx:
            for tid in self.name_compact_idx[s1_rec['name_compact']]:
                cand_scores[tid] += 8

        # Pass 4: Distinctive name tokens (IDF-weighted)
        for w in s1_rec['name_tokens']:
            df = self.token_df.get(('n', w), 0)
            if 0 < df <= self.df_cap:
                weight = 4 if df <= 50 else (2 if df <= 300 else 1)
                for tid in self.name_token_idx[w]:
                    cand_scores[tid] += weight

        # Pass 5: Character 3-gram retrieval (for spelling noise and OCR)
        fcn = s1_rec.get('name_norm_nosuffix', '')
        if fcn:
            for ng in set(get_char_ngrams(fcn, 3)):
                df = self.char_ngram_df.get(ng, 0)
                if 5 <= df <= 1500:
                    weight = 2 if df <= 100 else 1
                    for tid in self.name_char_ngram_idx[ng]:
                        cand_scores[tid] += weight

        # Pass 6 & 7: Address tokens & alphanumeric building anchors (IDF-weighted)
        for w in s1_rec['address_tokens']:
            df = self.token_df.get(('a', w), 0)
            if 0 < df <= self.df_cap:
                weight = 5 if df <= 30 else (3 if df <= 200 else 1)
                for tid in self.addr_token_idx[w]:
                    cand_scores[tid] += weight

        # Return top-K candidates
        return [tid for tid, _ in cand_scores.most_common(self.max_candidates)]
