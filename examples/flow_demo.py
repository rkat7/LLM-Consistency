#!/usr/bin/env python3
import sys
import os
import logging

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_consistency_verifier import ConsistencyVerifier
from llm_consistency_verifier.config.config import Config

def run_workflow_demo():
    """
    Demonstrate the enhanced workflow with detailed logging.
    """
    # Configure logging to see the detailed flow
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    print("\nLLM Consistency Verifier Workflow Demonstration")
    print("==============================================\n")
    
    # Initialize the verifier
    verifier = ConsistencyVerifier()
    
    # Example 1: Consistent text
    print("\n----- EXAMPLE 1: CONSISTENT TEXT -----\n")
    consistent_text = """
    Cats are mammals. All mammals have fur. Cats have fur.
    Some cats are black. Some cats are white. Different cats can have different colors.
    """
    
    print(f"Text: {consistent_text.strip()}\n")
    print("RUNNING VERIFICATION...\n")
    
    result = verifier.verify(consistent_text)
    
    print(f"\nVerification Result: {'CONSISTENT' if result.is_consistent else 'INCONSISTENT'}")
    print(f"Verification Time: {result.verification_time:.2f} seconds\n")
    
    # Example 2: Inconsistent text
    print("\n----- EXAMPLE 2: INCONSISTENT TEXT -----\n")
    inconsistent_text = """
    All birds can fly. Penguins are birds. Penguins cannot fly.
    """
    
    print(f"Text: {inconsistent_text.strip()}\n")
    print("RUNNING VERIFICATION...\n")
    
    result = verifier.verify(inconsistent_text)
    
    print(f"\nVerification Result: {'CONSISTENT' if result.is_consistent else 'INCONSISTENT'}")
    if not result.is_consistent:
        print("Inconsistencies:")
        for i, inconsistency in enumerate(result.inconsistencies, 1):
            print(f"  {i}. {inconsistency}")
    print(f"Verification Time: {result.verification_time:.2f} seconds\n")
    
    # Example 3: Repairing inconsistent text
    print("\n----- EXAMPLE 3: REPAIRING INCONSISTENT TEXT -----\n")
    print(f"Original text: {inconsistent_text.strip()}\n")
    print("RUNNING REPAIR...\n")
    
    repaired_text = verifier.repair(inconsistent_text)
    
    print(f"\nRepaired text: {repaired_text.strip()}\n")
    
    # Verify the repaired text
    print("VERIFYING REPAIRED TEXT...\n")
    result = verifier.verify(repaired_text)
    
    print(f"\nIs consistent after repair: {'YES' if result.is_consistent else 'NO'}")
    print(f"Verification Time: {result.verification_time:.2f} seconds\n")
    
    # Display the complete workflow
    print("\n----- COMPLETE WORKFLOW SUMMARY -----\n")
    print("""
    Text Input → Rule Extraction → Formalization → Verification → Results → Repair
    
    1. Text Input: Natural language text is received
    2. Rule Extraction: Logical rules are extracted using patterns or LLM
    3. Formalization: Rules are converted to formal logical representations
    4. Verification: Z3 solver checks for consistency
    5. Results: Consistency status and inconsistencies are reported
    6. Repair (Optional): LLM attempts to fix inconsistencies
    """)

if __name__ == "__main__":
    run_workflow_demo() 