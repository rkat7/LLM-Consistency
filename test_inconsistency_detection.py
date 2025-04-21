#!/usr/bin/env python3
"""
Test script for inconsistency detection in the LLM Consistency Verifier
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Ensure the current directory is in the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Load environment variables from .env
load_dotenv()

# Import directly from the module files
from llm_consistency_verifier.core.consistency_verifier import ConsistencyVerifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_classic_inconsistency():
    """Test the classic birds-penguins inconsistency"""
    verifier = ConsistencyVerifier()  # Use the default extractor
    
    # Test with a text containing a classic inconsistency
    text = """
    All birds can fly. Penguins are birds. Penguins cannot fly.
    """
    
    print("\n=== Testing Classic Inconsistency ===")
    verification_result = verifier.verify(text)
    
    print(f"Verification result: {'Consistent' if verification_result.is_consistent else 'Inconsistent'}")
    if not verification_result.is_consistent:
        print("Inconsistencies detected:")
        for i, inconsistency in enumerate(verification_result.inconsistencies, 1):
            print(f"{i}. {inconsistency}")
    
    # This should definitely be inconsistent
    return not verification_result.is_consistent

def test_complex_inconsistency():
    """Test a more complex inconsistency"""
    verifier = ConsistencyVerifier()  # Use the default extractor
    
    # Test with a text containing a more complex inconsistency
    text = """
    All philosophers are wise.
    All wise people are respected.
    Socrates is a philosopher.
    Socrates is not respected.
    """
    
    print("\n=== Testing Complex Transitive Inconsistency ===")
    verification_result = verifier.verify(text)
    
    print(f"Verification result: {'Consistent' if verification_result.is_consistent else 'Inconsistent'}")
    if not verification_result.is_consistent:
        print("Inconsistencies detected:")
        for i, inconsistency in enumerate(verification_result.inconsistencies, 1):
            print(f"{i}. {inconsistency}")
    
    # This should be inconsistent
    return not verification_result.is_consistent

def test_consistent_text():
    """Test a consistent text"""
    verifier = ConsistencyVerifier()
    
    # Test with a text that is consistent
    text = """
    All mammals are animals.
    Some animals can swim.
    Cats are mammals.
    """
    
    print("\n=== Testing Consistent Text ===")
    verification_result = verifier.verify(text)
    
    print(f"Verification result: {'Consistent' if verification_result.is_consistent else 'Inconsistent'}")
    if not verification_result.is_consistent:
        print("Inconsistencies detected:")
        for i, inconsistency in enumerate(verification_result.inconsistencies, 1):
            print(f"{i}. {inconsistency}")
    
    # This should be consistent
    return verification_result.is_consistent

def test_repair_function():
    """Test the repair functionality for inconsistent text"""
    verifier = ConsistencyVerifier()
    
    # Test with a text containing an inconsistency
    text = """
    All birds can fly. Penguins are birds. Penguins cannot fly.
    """
    
    print("\n=== Testing Repair Function ===")
    print("Original text:")
    print(text)
    
    # Repair the text
    repaired_text = verifier.repair(text)
    
    print("\nRepaired text:")
    print(repaired_text)
    
    # Verify the repaired text
    print("\nVerifying repaired text...")
    verification_result = verifier.verify(repaired_text)
    
    print(f"Verification result: {'Consistent' if verification_result.is_consistent else 'Inconsistent'}")
    if not verification_result.is_consistent:
        print("Remaining inconsistencies:")
        for i, inconsistency in enumerate(verification_result.inconsistencies, 1):
            print(f"{i}. {inconsistency}")
    
    # The repaired text should be consistent or improved
    return verification_result.is_consistent or (not verification_result.is_consistent and len(verification_result.inconsistencies) < 2)

def main():
    """Main test function"""
    print("Testing LLM Consistency Verifier...")
    
    # Run the tests
    tests = [
        test_classic_inconsistency,
        test_complex_inconsistency,
        test_consistent_text,
        test_repair_function
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