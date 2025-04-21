#!/usr/bin/env python3
"""
Test script for the LLM Interface
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables from .env
load_dotenv()

from llm_consistency_verifier.utils.llm_interface import LLMInterface
from llm_consistency_verifier.config.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_rule_extraction():
    """Test the rule extraction functionality"""
    llm = LLMInterface()
    
    # Test with a simple text containing logical rules
    text = """
    All birds have wings. Some birds can't fly.
    If an animal is a mammal, then it has fur.
    Cats are mammals. Dogs are mammals.
    Fish are not mammals.
    """
    
    print("\n=== Testing Rule Extraction ===")
    rules = llm.extract_logical_rules(text)
    
    # Display the extracted rules
    print(f"Extracted {len(rules)} rules:")
    for i, rule in enumerate(rules, 1):
        rule_type = rule['type']
        if rule_type == 'universal':
            print(f"{i}. Universal: {rule.get('statement')}")
        elif rule_type == 'existential':
            print(f"{i}. Existential: {rule.get('statement')}")
        elif rule_type == 'implication':
            print(f"{i}. Implication: If {rule.get('antecedent')} then {rule.get('consequent')}")
        elif rule_type == 'negation':
            print(f"{i}. Negation: Not({rule.get('statement')})")
        elif rule_type == 'assertion':
            print(f"{i}. Assertion: {rule.get('statement')}")
    
    return len(rules) > 0

def test_inconsistency_repair():
    """Test the inconsistency repair functionality"""
    llm = LLMInterface()
    
    # Test with a text containing inconsistencies
    text = """
    All birds can fly. Penguins are birds. Penguins cannot fly.
    """
    
    # List of identified inconsistencies
    inconsistencies = [
        "Text states 'All birds can fly' but also states 'Penguins are birds' and 'Penguins cannot fly', which is inconsistent."
    ]
    
    print("\n=== Testing Inconsistency Repair ===")
    repaired_text = llm.repair_inconsistencies(text, inconsistencies)
    
    # Display the repaired text
    print("Original text:")
    print(text)
    print("\nRepaired text:")
    print(repaired_text)
    
    return repaired_text != text

def test_error_handling():
    """Test error handling in the LLM interface"""
    llm = LLMInterface()
    
    print("\n=== Testing Error Handling ===")
    
    # Test with empty text
    rules = llm.extract_logical_rules("")
    print(f"Empty text extraction returned {len(rules)} rules (should be 0)")
    
    # Test with None
    rules = llm.extract_logical_rules(None)
    print(f"None text extraction returned {len(rules)} rules (should be 0)")
    
    # Test repair with empty inconsistencies
    repaired = llm.repair_inconsistencies("Some text", [])
    print(f"Empty inconsistencies repair returned text of length {len(repaired)}")
    
    return True

def main():
    """Main test function"""
    print("Testing LLM Interface...")
    
    # Check if the required API key is set
    if not Config.OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY is not set. Please set it in your environment variables or .env file.")
        return False
    
    # Run the tests
    tests = [
        test_rule_extraction,
        test_inconsistency_repair,
        test_error_handling
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print(f"{test.__name__}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            print(f"{test.__name__}: ERROR - {str(e)}")
            results.append(False)
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"Tests passed: {sum(results)}/{len(tests)}")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 