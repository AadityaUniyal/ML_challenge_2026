import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.normalization import normalize_record
from src.features import extract_pair_features

class TestFeatures(unittest.TestCase):
    def test_feature_vector_dimension(self):
        s1 = normalize_record("S1-01", "Walmart Supercenter", "100 Main St, Dallas, TX", "US")
        s2 = normalize_record("S2-02", "Walmart Inc", "100 Main Street, Dallas", "US")
        feats = extract_pair_features(s1, s2, 12.0)
        self.assertEqual(len(feats), 15)
        # Token set ratio handles common prefixes / subsets
        self.assertGreater(feats[3], 0.70)
        # No number conflict (both 100)
        self.assertEqual(feats[10], 0.0)

    def test_number_conflict(self):
        s1 = normalize_record("S1-01", "Best Buy", "500 Lake Rd", "US")
        s2 = normalize_record("S2-02", "Best Buy", "999 Oak St", "US")
        feats = extract_pair_features(s1, s2, 5.0)
        self.assertEqual(feats[10], 1.0)    # Has conflict = 1

if __name__ == '__main__':
    unittest.main()
