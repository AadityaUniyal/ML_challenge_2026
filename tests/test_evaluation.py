import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation import compute_entity_f05, evaluate_macro_f05

class TestEvaluation(unittest.TestCase):
    def test_official_example(self):
        pred = {"S2-00047", "S2-00193", "S3-00812"}
        true = {"S2-00047", "S3-00812"}
        score = compute_entity_f05(pred, true)
        self.assertAlmostEqual(score, 0.7142857, places=4)

    def test_singleton_scoring(self):
        # Empty prediction for singleton = 1.0
        self.assertEqual(compute_entity_f05(set(), set()), 1.0)
        # False merge on singleton = 0.0
        self.assertEqual(compute_entity_f05({"S2-99"}, set()), 0.0)

if __name__ == '__main__':
    unittest.main()
