import sys
import os
import unittest

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_consistency_verifier.core.verification_engine import VerificationEngine, VerificationResult
from llm_consistency_verifier.core.consistency_verifier import ConsistencyVerifier
from llm_consistency_verifier.utils.llm_interface import LLMInterface

class TestVerificationEngine(unittest.TestCase):
    """Test cases for the verification engine."""
    
    def setUp(self):
        """Set up the test environment."""
        self.engine = VerificationEngine(solver_type="z3")
    
    def test_simple_consistent_rules(self):
        """Test verification of simple consistent logical rules."""
        rules = [
            {
                "type": "implication",
                "antecedent": "is_cat",
                "consequent": "is_mammal",
                "original_text": "If it is a cat, then it is a mammal"
            },
            {
                "type": "implication",
                "antecedent": "is_mammal",
                "consequent": "has_fur",
                "original_text": "If it is a mammal, then it has fur"
            },
            {
                "type": "assertion",
                "statement": "is_cat",
                "original_text": "It is a cat"
            }
        ]
        
        result = self.engine.verify(rules)
        self.assertTrue(result.is_consistent)
        self.assertEqual(len(result.inconsistencies), 0)
    
    def test_simple_inconsistent_rules(self):
        """Test verification of simple inconsistent logical rules."""
        rules = [
            {
                "type": "implication",
                "antecedent": "is_bird",
                "consequent": "can_fly",
                "original_text": "All birds can fly"
            },
            {
                "type": "assertion",
                "statement": "is_bird",
                "original_text": "Penguins are birds"
            },
            {
                "type": "negation",
                "statement": "can_fly",
                "original_text": "Penguins cannot fly"
            }
        ]
        
        result = self.engine.verify(rules)
        self.assertFalse(result.is_consistent)
        self.assertGreater(len(result.inconsistencies), 0)
    
    def test_contradiction_detection(self):
        """Test detection of direct contradictions."""
        rules = [
            {
                "type": "assertion",
                "statement": "is_red",
                "original_text": "The ball is red"
            },
            {
                "type": "negation",
                "statement": "is_red",
                "original_text": "The ball is not red"
            }
        ]
        
        result = self.engine.verify(rules)
        self.assertFalse(result.is_consistent)
        self.assertGreater(len(result.inconsistencies), 0)
        
        # Check if the contradiction message contains the original text
        self.assertTrue(any("ball is red" in inc for inc in result.inconsistencies))

class TestConsistencyVerifier(unittest.TestCase):
    """Test cases for the consistency verifier."""
    
    def setUp(self):
        """Set up the test environment."""
        # Skip these tests if running in an environment without an LLM API key
        if not os.environ.get('OPENAI_API_KEY'):
            self.skipTest("Skipping test that requires API key")
        
        self.verifier = ConsistencyVerifier()
    
    def test_consistent_text(self):
        """Test verification of consistent text."""
        text = "Cats are mammals. All mammals have fur. Cats have fur."
        result = self.verifier.verify(text)
        self.assertTrue(result.is_consistent)
    
    def test_inconsistent_text(self):
        """Test verification of inconsistent text."""
        text = "All birds can fly. Penguins are birds. Penguins cannot fly."
        result = self.verifier.verify(text)
        self.assertFalse(result.is_consistent)
    
    def test_repair(self):
        """Test repairing inconsistent text."""
        text = "All birds can fly. Penguins are birds. Penguins cannot fly."
        repaired_text = self.verifier.repair(text)
        
        # Check that the repaired text is different from the original
        self.assertNotEqual(repaired_text, text)
        
        # Verify that the repaired text is consistent
        # Note: This may fail if the repair is unsuccessful
        repair_result = self.verifier.verify(repaired_text)
        self.assertTrue(repair_result.is_consistent)
    
    def test_explain_inconsistencies(self):
        """Test explanation of inconsistencies."""
        text = "All birds can fly. Penguins are birds. Penguins cannot fly."
        result = self.verifier.verify(text)
        
        explanation = self.verifier.explain_inconsistencies(result)
        
        # Check that the explanation is not empty
        self.assertTrue(len(explanation) > 0)
        
        # Check that the explanation mentions inconsistencies
        self.assertIn("inconsistencies", explanation.lower())

if __name__ == '__main__':
    unittest.main() 