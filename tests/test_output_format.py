import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestOutputFormat(unittest.TestCase):
    def test_output_files_exist_and_headers(self):
        match_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "matching_results.tsv")
        cand_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "candidate_pairs.tsv")
        
        self.assertTrue(os.path.isfile(match_file), f"Missing {match_file}")
        self.assertTrue(os.path.isfile(cand_file), f"Missing {cand_file}")
        
        with open(match_file, 'r', encoding='utf-8') as f:
            header = f.readline().rstrip('\r\n')
            self.assertEqual(header, "source1_entity_id\tmatched_entity_ids")
            
        with open(cand_file, 'r', encoding='utf-8') as f:
            header = f.readline().rstrip('\r\n')
            self.assertEqual(header, "source1_entity_id\tcandidate_entity_ids")

if __name__ == '__main__':
    unittest.main()
