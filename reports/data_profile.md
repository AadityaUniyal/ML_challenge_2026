# Dataset Profile Report — Amazon ML Challenge 2026

## 1. Overview
- **Train Source 1 (Reference):** 2,206,821 rows | Countries: {'US': 1323633, 'India': 883188}
- **Train Source 2:** 5,034,616 rows | Countries: {'India': 2017799, 'US': 3016817}
- **Train Source 3:** 5,285,603 rows | Countries: {'US': 3170056, 'India': 2115547}
- **Test Source 1:** 1,732,544 rows | Countries: {'US': 663106, 'France': 259452, 'India': 809986}
- **Test Source 2:** 4,887,273 rows | Countries: {'India': 2312565, 'France': 703378, 'US': 1871330}
- **Test Source 3:** 5,082,316 rows | Countries: {'India': 2405000, 'France': 731615, 'US': 1945701}

## 2. Key Findings
- **Zero Cross-Country Matches:** 100% of entity matches occur within the same country.
- **Unseen Country in Test:** Test data introduces **France** (259k entities in S1, 1.43M in S2/S3).
- **Singletons:** ~5.58% of reference entities have zero matches and must be predicted empty.
