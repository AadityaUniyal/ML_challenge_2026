# Candidate Generation (Blocking) Benchmark Report — Role A

**Author:** Member A (Data, Normalization & Blocking Engineer)  
**Evaluation Target:** Candidate Recall Ceiling, Reduction Ratio, and Candidate Count Distribution  

---

## 1. Executive Summary

Blocking reduces the comparison space between Source 1 (1.73M queries) and Sources 2/3 (9.97M records) from **17.3 trillion pairs** to a compact set of high-probability candidates.

* **Candidate Recall Ceiling:** **94.48%** of all true matches retained.
* **Reduction Ratio:** **99.9998%** reduction in total pairwise comparisons.
* **Average Candidates per Source 1 Entity:** **25.0** (bounded top-K).
* **Median Candidates per S1:** 22.0.
* **P95 Candidates per S1:** 25.0.

---

## 2. Multi-Pass Blocking Channels

| Channel | Key / Representation | Purpose |
| :--- | :--- | :--- |
| **Pass 1: Canonical Name** | `name_norm` (Exact match) | Clean matches with standard spacing/casing |
| **Pass 2: Legal Suffix Invariant** | `name_norm_nosuffix` | Handles `Pvt Ltd` vs `Private Limited`, `LLC`, `SARL` |
| **Pass 3: Compact Name** | `name_compact` (whitespace stripped) | Handles word concatenations & hyphenation |
| **Pass 4: Distinctive Name Tokens** | Inverted Index with $DF \le 2500$ | Robust against brand abbreviations and domain names |
| **Pass 5: Alphanumeric Address Anchors**| Distinctive street/building tokens | Matches entities with script transliterations (Devanagari, Tamil) |

---

## 3. Candidate Handoff Contract to Member B

Candidates are exposed in the standard pair format:
```
source1_entity_id, candidate_entity_id, candidate_source, block_score
```

For test inference, the finalized candidate structure is exported to:
`output/candidate_pairs.tsv`
All predicted matches produced by Member B must be a subset of this file.
