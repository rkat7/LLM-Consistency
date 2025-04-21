#!/usr/bin/env python3
import sys
from llm_consistency_verifier.core.consistency_verifier import ConsistencyVerifier

def test_manual_input():
    # Initialize the verifier
    verifier = ConsistencyVerifier()
    
    # Get input text
    if len(sys.argv) > 1:
        # Use command line argument if provided
        input_text = sys.argv[1]
    else:
        # Otherwise prompt the user
        print("\nEnter text to verify logical consistency (press Enter twice to finish):")
        lines = []
        while True:
            line = input()
            if not line and lines and not lines[-1]:
                # Two consecutive empty lines
                break
            lines.append(line)
        input_text = "\n".join(lines[:-1])  # Remove the last empty line
    
    print(f"\n=== Input Text ===\n{input_text}\n")
    
    # Verify consistency
    print("=== Verifying Consistency ===")
    result = verifier.verify(input_text)
    
    # Show results
    print("\n=== Results ===")
    print(f"Is consistent: {result.is_consistent}")
    
    if not result.is_consistent:
        print("\nInconsistencies detected:")
        for i, inconsistency in enumerate(result.inconsistencies, 1):
            print(f"{i}. {inconsistency}")
    
    # Try repair if inconsistent
    if not result.is_consistent:
        print("\n=== Attempting Repair ===")
        repaired_text = verifier.repair(input_text)
        
        # Verify the repaired text
        print("\n=== Repaired Text ===")
        print(repaired_text)
        
        repaired_result = verifier.verify(repaired_text)
        print(f"\nRepaired text is consistent: {repaired_result.is_consistent}")

if __name__ == "__main__":
    test_manual_input() 