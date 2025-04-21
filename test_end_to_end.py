#!/usr/bin/env python3
"""
End-to-end test for the LLM Consistency Verifier system.
This tests the entire pipeline from text input to inconsistency detection to repair.
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Ensure the current directory is in the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Load environment variables from .env
load_dotenv()

from llm_consistency_verifier import ConsistencyVerifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_case(verifier, title, text, should_be_inconsistent=True):
    """Test a specific case with the consistency verifier"""
    print(f"\n\n{'=' * 80}")
    print(f"TEST CASE: {title}")
    print(f"{'=' * 80}")
    print(f"INPUT TEXT:\n{text}\n")
    
    # Verify consistency
    print("VERIFICATION RESULTS:")
    result = verifier.verify(text)
    
    print(f"Is consistent: {result.is_consistent}")
    if not result.is_consistent:
        print("Inconsistencies detected:")
        for i, inconsistency in enumerate(result.inconsistencies, 1):
            print(f"{i}. {inconsistency}")
    
    # Check if the result matches expectations
    if (not result.is_consistent) == should_be_inconsistent:
        print("\n✅ TEST PASSED: Inconsistency detection worked as expected")
    else:
        print("\n❌ TEST FAILED: Expected inconsistent" if should_be_inconsistent else "\n❌ TEST FAILED: Expected consistent")
    
    # If inconsistent, try to repair
    if not result.is_consistent:
        print("\nREPAIR RESULTS:")
        repaired_text = verifier.repair(text)
        print(f"Repaired text:\n{repaired_text}\n")
        
        # Verify the repaired text
        repaired_result = verifier.verify(repaired_text)
        print(f"Repaired text is consistent: {repaired_result.is_consistent}")
        
        if repaired_result.is_consistent:
            print("✅ REPAIR SUCCESSFUL: Fixed all inconsistencies")
        else:
            print("❌ REPAIR FAILED: Some inconsistencies remain")
            for i, inconsistency in enumerate(repaired_result.inconsistencies, 1):
                print(f"{i}. {inconsistency}")
    
    return (not result.is_consistent) == should_be_inconsistent

def main():
    """Main test function for end-to-end testing"""
    print("Starting end-to-end tests of LLM Consistency Verifier...")
    
    # Initialize the verifier
    verifier = ConsistencyVerifier()
    
    # Define test cases with expected outcomes
    test_cases = [
        # Basic inconsistency cases
        ("Classic Birds-Penguins", """
        All birds can fly. Penguins are birds. Penguins cannot fly.
        """, True),
        
        ("Transitivity Inconsistency", """
        All philosophers are wise.
        All wise people are respected.
        Socrates is a philosopher.
        Socrates is not respected.
        """, True),
        
        # Consistent cases
        ("Simple Consistent Text", """
        All mammals are animals.
        Some animals can swim.
        Cats are mammals.
        """, False),
        
        ("Exception-Based Consistent", """
        Most birds can fly. Penguins are birds. Penguins cannot fly.
        """, False),
        
        # More complex cases
        ("Complex Domain Knowledge", """
        All programming languages have syntax.
        Python is a programming language.
        Python uses indentation for block structure.
        Java uses braces for block structure.
        Python does not use braces for block structure.
        """, False),
        
        ("Negation Inconsistency", """
        No planets are stars.
        Earth is a planet.
        Earth is a star.
        """, True)
    ]
    
    # Run all test cases
    results = []
    for title, text, should_be_inconsistent in test_cases:
        results.append(test_case(verifier, title, text, should_be_inconsistent))
    
    # Print summary
    print("\n\n" + "=" * 80)
    print("END-TO-END TEST SUMMARY")
    print("=" * 80)
    print(f"Total test cases: {len(results)}")
    print(f"Tests passed: {sum(results)}")
    print(f"Tests failed: {len(results) - sum(results)}")
    
    if sum(results) == len(results):
        print("\n✅ ALL TESTS PASSED!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 