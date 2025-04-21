#!/usr/bin/env python3
import sys
import time
import logging
from llm_consistency_verifier.core.consistency_verifier import ConsistencyVerifier
from llm_consistency_verifier.core.enhanced_consistency_verifier import EnhancedConsistencyVerifier
from llm_consistency_verifier.config.config import Config
import traceback

# Configure logging to see the process
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_result(result, case_title):
    """Format the verification result for display."""
    if result is None:
        return f"\n{'-' * 80}\nResult for: {case_title}\nStatus: ❌ ERROR (verification failed)\n"
    
    is_consistent = result.is_consistent
    status = "✅ CONSISTENT" if is_consistent else "❌ INCONSISTENT"
    
    output = f"\n{'-' * 80}\n"
    output += f"Result for: {case_title}\n"
    output += f"Status: {status}\n"
    output += f"Verification time: {result.verification_time:.4f} seconds\n"
    
    if not is_consistent:
        output += "Inconsistencies:\n"
        for i, inconsistency in enumerate(result.inconsistencies, 1):
            output += f"  {i}. {inconsistency}\n"
    
    return output

def compare_verifiers(text, title, timeout=120):
    """Compare standard and enhanced verifiers on the same input."""
    print(f"\n{'=' * 80}")
    print(f"TEST CASE: {title}")
    print(f"{'=' * 80}")
    print(f"Input text: {text}")
    
    standard_result = None
    enhanced_result = None
    
    # Standard verifier with timeout and error handling
    try:
        standard_verifier = ConsistencyVerifier()
        start_time = time.time()
        
        # Use a timeout to prevent hanging
        max_time = start_time + timeout
        standard_result = standard_verifier.verify(text)
        standard_result.verification_time = time.time() - start_time
        
        print("\n--- STANDARD VERIFIER ---")
        print(format_result(standard_result, title))
    except Exception as e:
        print(f"\n--- STANDARD VERIFIER ERROR ---")
        print(f"Error: {str(e)}")
        traceback.print_exc()
    
    # Wait between API calls to avoid rate limiting
    time.sleep(5)
    
    # Enhanced verifier with timeout and error handling
    try:
        enhanced_verifier = EnhancedConsistencyVerifier()
        start_time = time.time()
        
        # Use a timeout to prevent hanging
        max_time = start_time + timeout
        enhanced_result = enhanced_verifier.verify(text)
        enhanced_result.verification_time = time.time() - start_time
        
        print("\n--- ENHANCED VERIFIER ---")
        print(format_result(enhanced_result, title))
    except Exception as e:
        print(f"\n--- ENHANCED VERIFIER ERROR ---")
        print(f"Error: {str(e)}")
        traceback.print_exc()
    
    # Analyze differences if both verifiers worked
    if standard_result is not None and enhanced_result is not None:
        if standard_result.is_consistent != enhanced_result.is_consistent:
            print("\n⚠️  DIFFERENT RESULTS: Standard and enhanced verifiers disagree!")
            
            if standard_result.is_consistent:
                print("Standard verifier found no inconsistencies, but enhanced verifier did.")
            else:
                print("Standard verifier found inconsistencies, but enhanced verifier did not.")
        else:
            print("\n✓ AGREEMENT: Both verifiers produced the same consistency result.")
    
    return standard_result, enhanced_result

def test_basic_cases():
    """Test cases with basic logical relations."""
    print("\n" + "=" * 80)
    print("BASIC LOGICAL RELATIONSHIP TESTS")
    print("=" * 80)
    
    # Test Case 1: Classic birds-penguin inconsistency
    birds_penguin_text = "All birds can fly. Penguins are birds. Penguins cannot fly."
    compare_verifiers(birds_penguin_text, "Classic Birds-Penguin Inconsistency")
    
    # Test Case 2: Consistent case
    consistent_text = "All birds have feathers. Penguins are birds. Penguins have feathers."
    compare_verifiers(consistent_text, "Consistent Birds-Feathers Case")
    
    # Test Case 3: Simple direct contradiction
    direct_contradiction = "Cats are mammals. Cats are not mammals."
    compare_verifiers(direct_contradiction, "Direct Contradiction")

def test_selected_advanced_cases():
    """Test a selection of more complex logical relations to avoid rate limiting."""
    print("\n" + "=" * 80)
    print("SELECTED ADVANCED LOGICAL RELATIONSHIP TESTS")
    print("=" * 80)
    
    # Test Case 4: Hierarchical inheritance - most complex and useful case
    hierarchy_text = """
    All animals are living things.
    All mammals are animals.
    All cats are mammals.
    Fluffy is a cat.
    Fluffy is not a living thing.
    """
    compare_verifiers(hierarchy_text.strip(), "Deep Hierarchical Inheritance")
    
    # Test Case 5: Complex conditional statements - most useful for Z3 testing
    complex_conditional = """
    If it rains, then the ground gets wet.
    If the ground is wet, then the grass is slippery.
    If the grass is slippery, then it is dangerous to run.
    It rained yesterday.
    It was safe to run on the grass yesterday.
    """
    compare_verifiers(complex_conditional.strip(), "Complex Conditional Chain")

def test_repair_functionality():
    """Test the repair functionality of both verifiers."""
    print("\n" + "=" * 80)
    print("REPAIR FUNCTIONALITY TEST")
    print("=" * 80)
    
    # Test with the birds-penguin case
    text = "All birds can fly. Penguins are birds. Penguins cannot fly."
    
    try:
        # Standard verifier repair
        standard_verifier = ConsistencyVerifier()
        print("Standard Verifier Repair:")
        result = standard_verifier.verify(text)
        if not result.is_consistent:
            repaired_text = standard_verifier.repair(text)
            print(f"Original: {text}")
            print(f"Repaired: {repaired_text}")
            repair_result = standard_verifier.verify(repaired_text)
            print(f"Repair successful: {repair_result.is_consistent}")
        else:
            print("No repair needed (text is consistent)")
    except Exception as e:
        print(f"Standard verifier repair error: {str(e)}")
    
    # Wait between API calls to avoid rate limiting
    time.sleep(5)
    
    try:
        # Enhanced verifier repair
        enhanced_verifier = EnhancedConsistencyVerifier()
        print("\nEnhanced Verifier Repair:")
        result = enhanced_verifier.verify(text)
        if not result.is_consistent:
            repaired_text = enhanced_verifier.repair(text)
            print(f"Original: {text}")
            print(f"Repaired: {repaired_text}")
            repair_result = enhanced_verifier.verify(repaired_text)
            print(f"Repair successful: {repair_result.is_consistent}")
        else:
            print("No repair needed (text is consistent)")
    except Exception as e:
        print(f"Enhanced verifier repair error: {str(e)}")

def main():
    """Run the tests with error handling."""
    print("CONSISTENCY VERIFIER COMPARISON TESTS")
    print("Standard vs Enhanced Implementation")
    print("Running with error handling and API rate limiting precautions")
    
    try:
        # Run basic test cases
        test_basic_cases()
        
        # Wait before running more complex tests
        time.sleep(10)
        
        # Run selected advanced cases
        test_selected_advanced_cases()
        
        # Wait before testing repair functionality
        time.sleep(10)
        
        # Test repair functionality
        test_repair_functionality()
        
        print("\nAll tests completed!")
    except Exception as e:
        print(f"Error during test execution: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 