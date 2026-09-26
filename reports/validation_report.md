# Model Validation Report — Role B

**Author:** Member B (Matching Model, Features & Evaluation Engineer)  
**Target:** Pairwise Scoring, Macro $F_{0.5}$ Optimization & Error Profiling  

---

## 1. Validation Split Design

* **Methodology:** Stratified Entity-Level Hold-Out Split (80% Train, 20% Validation).
* **Entity Count:** 50,000 Train $S_1$ entities, 10,000 Out-of-Fold Validation $S_1$ entities.
* **Pairs Evaluated:** 1,249,234 candidate pairs (153,615 positive pairs, 1,095,619 hard negative pairs).

---

## 2. Feature Importance Breakdown

| Feature | Importance | Interpretation |
| :--- | :--- | :--- |
| `comb_set` | 993 | Combined Name + Address token set ratio |
| `cand_score` | 794 | Blocker retrieval prior score |
| `name_jw` | 785 | Jaro-Winkler name similarity |
| `addr_set` | 743 | Address token set ratio |
| `len_diff_addr` | 703 | Address length differential |
| `len_diff_name` | 692 | Name length differential |
| `name_partial` | 667 | Trade-name / DBA containment |
| `name_sort` | 594 | Word-order transposition tolerance |
| `addr_ratio` | 584 | Fine-grained address character similarity |
| `addr_sort` | 583 | Address component reordering |
| `diff_nums` | 494 | Address number symmetric difference |
| `name_set` | 480 | Legal suffix variation tolerance |
| `name_ratio` | 443 | Exact character sequence fidelity |
| `common_nums` | 264 | Shared building/plot number anchor |
| `has_conflict` | 181 | Conflict penalty for contradictory numbers |

---

## 3. Decision Threshold Optimization (Macro $F_{0.5}$)

$$\text{Macro } F_{0.5} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$$

* $\tau = 0.50 \implies F_{0.5} = 0.9153$
* $\tau = 0.55 \implies F_{0.5} = 0.9167$
* $\tau = 0.60 \implies F_{0.5} = 0.9185$
* **$\tau = 0.65 \implies F_{0.5} = \mathbf{0.9188}$** (Optimal)
* $\tau = 0.70 \implies F_{0.5} = 0.9183$
* $\tau = 0.80 \implies F_{0.5} = 0.9173$

**Conclusion:** Setting decision boundary $\tau = 0.65$ eliminates marginal false positives, maximizing the precision weight ($\beta = 0.5$) and correctly scoring 1.0 on singletons.
