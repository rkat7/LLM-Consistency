#!/usr/bin/env python3
import sys
import os

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_consistency_verifier import ConsistencyVerifier

def run_basic_example():
    """
    Run a basic example of the ConsistencyVerifier.
    """
    print("LLM Consistency Verifier Basic Example")
    print("======================================\n")
    
    # Initialize the verifier
    verifier = ConsistencyVerifier()
    
    # Example 1: Consistent text
    consistent_text = """
    Cats are mammals. All mammals have fur. Cats have fur.
    Some cats are black. Some cats are white. Different cats can have different colors.
    """
    
    print("Example 1: Checking consistent text")
    print(f"Text: {consistent_text.strip()}\n")
    
    result = verifier.verify(consistent_text)
    print(f"Result: {result.is_consistent}")
    print(f"Verification time: {result.verification_time:.2f} seconds\n")
    
    # Example 2: Inconsistent text
    inconsistent_text = """
    All birds can fly. Penguins are birds. Penguins cannot fly.
    """
    
    print("Example 2: Checking inconsistent text")
    print(f"Text: {inconsistent_text.strip()}\n")
    
    result = verifier.verify(inconsistent_text)
    print(f"Result: {result.is_consistent}")
    if not result.is_consistent:
        print("Inconsistencies:")
        for i, inconsistency in enumerate(result.inconsistencies, 1):
            print(f"  {i}. {inconsistency}")
    print(f"Verification time: {result.verification_time:.2f} seconds\n")
    
    # Example 3: Repairing inconsistent text
    print("Example 3: Repairing inconsistent text")
    print(f"Original text: {inconsistent_text.strip()}\n")
    
    repaired_text = verifier.repair(inconsistent_text)
    print(f"Repaired text: {repaired_text.strip()}\n")
    
    # Verify the repaired text
    result = verifier.verify(repaired_text)
    print(f"Is consistent after repair: {result.is_consistent}")
    print(f"Verification time: {result.verification_time:.2f} seconds\n")
    
    # Example 4: Complex inconsistent text
    complex_text = """
    If a shape is a square, then it is a rectangle.
    If a shape is a rectangle, then it has four sides.
    All squares have exactly four sides.
    Some shapes have four sides but are not rectangles.
    All rectangles are squares.
    """
    
    print("Example 4: Complex inconsistent text")
    print(f"Text: {complex_text.strip()}\n")
    
    result = verifier.verify(complex_text)
    print(f"Result: {result.is_consistent}")
    if not result.is_consistent:
        print("Inconsistencies:")
        for i, inconsistency in enumerate(result.inconsistencies, 1):
            print(f"  {i}. {inconsistency}")
            
    # Get explanation of inconsistencies
    explanation = verifier.explain_inconsistencies(result)
    print("\nExplanation:")
    print(explanation)
    
    print(f"Verification time: {result.verification_time:.2f} seconds\n")

if __name__ == "__main__":
    run_basic_example() 