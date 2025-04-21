#!/usr/bin/env python3
import os
import sys
import time
from llm_consistency_verifier.core.enhanced_consistency_verifier import EnhancedConsistencyVerifier
from llm_consistency_verifier.core.consistency_verifier import ConsistencyVerifier

# ANSI color codes for better output formatting
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}")

def print_success(text):
    print(f"{Colors.GREEN}{text}{Colors.ENDC}")

def print_warning(text):
    print(f"{Colors.YELLOW}{text}{Colors.ENDC}")

def print_error(text):
    print(f"{Colors.RED}{text}{Colors.ENDC}")

def print_info(text):
    print(f"{Colors.BLUE}{text}{Colors.ENDC}")

def get_multiline_input():
    print_info("Enter your text for logical consistency verification.")
    print_info("Enter a blank line when you are done typing:")
    print("-------------------------------------------")

    lines = []
    while True:
        try:
            line = input()
            if line.strip() == '':
                break
            lines.append(line)
        except EOFError:
            break

    return '\n'.join(lines)

def verify_text(text, use_enhanced=True):
    print_header("VERIFICATION PROCESS")
    print_info("Verifying logical consistency...")
    
    start_time = time.time()
    
    if use_enhanced:
        verifier = EnhancedConsistencyVerifier()
        print_info("Using Enhanced Consistency Verifier")
    else:
        verifier = ConsistencyVerifier()
        print_info("Using Standard Consistency Verifier")
    
    result = verifier.verify(text)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print_header("RESULTS")
    
    if result.is_consistent:
        print_success(f"✓ CONSISTENT (verified in {duration:.2f} seconds)")
    else:
        print_error(f"✗ INCONSISTENT (verified in {duration:.2f} seconds)")
        
        print_header("INCONSISTENCIES")
        for i, inconsistency in enumerate(result.inconsistencies, 1):
            print_error(f"{i}. {inconsistency}")
    
    return result, verifier

def repair_text(text, verifier):
    print_header("REPAIR PROCESS")
    print_info("Repairing inconsistencies...")
    
    start_time = time.time()
    repaired_text = verifier.repair(text)
    end_time = time.time()
    
    print_info(f"Repair completed in {end_time - start_time:.2f} seconds")
    
    print_header("REPAIRED TEXT")
    print(repaired_text)
    
    return repaired_text

def compare_verifiers(text):
    print_header("COMPARISON MODE")
    print_info("Running both verifiers for comparison...")
    
    # Run standard verifier
    print_header("STANDARD VERIFIER")
    standard_verifier = ConsistencyVerifier()
    start_time = time.time()
    standard_result = standard_verifier.verify(text)
    standard_time = time.time() - start_time
    
    if standard_result.is_consistent:
        print_success(f"✓ CONSISTENT (verified in {standard_time:.2f} seconds)")
    else:
        print_error(f"✗ INCONSISTENT (verified in {standard_time:.2f} seconds)")
        print_header("STANDARD VERIFIER INCONSISTENCIES")
        for i, inconsistency in enumerate(standard_result.inconsistencies, 1):
            print_error(f"{i}. {inconsistency}")
    
    # Run enhanced verifier
    print_header("ENHANCED VERIFIER")
    enhanced_verifier = EnhancedConsistencyVerifier()
    start_time = time.time()
    enhanced_result = enhanced_verifier.verify(text)
    enhanced_time = time.time() - start_time
    
    if enhanced_result.is_consistent:
        print_success(f"✓ CONSISTENT (verified in {enhanced_time:.2f} seconds)")
    else:
        print_error(f"✗ INCONSISTENT (verified in {enhanced_time:.2f} seconds)")
        print_header("ENHANCED VERIFIER INCONSISTENCIES")
        for i, inconsistency in enumerate(enhanced_result.inconsistencies, 1):
            print_error(f"{i}. {inconsistency}")
    
    # Compare results
    print_header("COMPARISON SUMMARY")
    if standard_result.is_consistent == enhanced_result.is_consistent:
        print_success("Both verifiers agree on the consistency status")
    else:
        print_warning("Verifiers disagree on the consistency status")
    
    print_info(f"Standard verifier: {standard_time:.2f} seconds")
    print_info(f"Enhanced verifier: {enhanced_time:.2f} seconds")
    print_info(f"Speedup/slowdown: {standard_time/enhanced_time:.2f}x")
    
    return standard_result, enhanced_result

def show_menu():
    print_header("LLM CONSISTENCY VERIFIER")
    print("1. Verify text with Enhanced Verifier")
    print("2. Verify text with Standard Verifier")
    print("3. Compare both verifiers")
    print("4. Load example text")
    print("5. Exit")
    
    choice = input("\nEnter your choice (1-5): ")
    return choice

def load_example():
    examples = {
        "1": "All birds have feathers. Penguins are birds. Penguins have feathers.",
        "2": "All birds can fly. Penguins are birds. Penguins cannot fly.",
        "3": """All animals are living things.
All mammals are animals.
All cats are mammals.
Fluffy is a cat.
Fluffy is not a living thing.""",
        "4": """If it rains, then the ground gets wet.
If the ground is wet, then the grass is slippery.
If the grass is slippery, then it is dangerous to run.
It rained yesterday.
It was safe to run on the grass yesterday."""
    }
    
    print_header("EXAMPLE TEXTS")
    print("1. Consistent birds-feathers example")
    print("2. Inconsistent birds-fly example")
    print("3. Hierarchical inheritance example")
    print("4. Complex conditional chain example")
    
    choice = input("\nSelect an example (1-4): ")
    
    if choice in examples:
        return examples[choice]
    else:
        print_error("Invalid choice. Using default example.")
        return examples["1"]

def main():
    # Check for OpenAI API key
    if "OPENAI_API_KEY" not in os.environ:
        print_error("ERROR: OPENAI_API_KEY environment variable not found!")
        print_info("Please set your OpenAI API key before running this script:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    while True:
        choice = show_menu()
        
        if choice == "1":
            text = get_multiline_input()
            result, verifier = verify_text(text, use_enhanced=True)
            
            if not result.is_consistent:
                repair_choice = input("\nWould you like to repair the inconsistencies? (y/n): ").lower()
                if repair_choice == 'y':
                    repair_text(text, verifier)
        
        elif choice == "2":
            text = get_multiline_input()
            result, verifier = verify_text(text, use_enhanced=False)
            
            if not result.is_consistent:
                repair_choice = input("\nWould you like to repair the inconsistencies? (y/n): ").lower()
                if repair_choice == 'y':
                    repair_text(text, verifier)
        
        elif choice == "3":
            text = get_multiline_input()
            standard_result, enhanced_result = compare_verifiers(text)
            
            if not enhanced_result.is_consistent:
                repair_choice = input("\nWould you like to repair the inconsistencies? (y/n): ").lower()
                if repair_choice == 'y':
                    verifier = EnhancedConsistencyVerifier()
                    repair_text(text, verifier)
        
        elif choice == "4":
            text = load_example()
            print_header("LOADED EXAMPLE")
            print(text)
            
            choice = input("\nWhich verifier would you like to use? (1=Enhanced, 2=Standard, 3=Compare): ")
            
            if choice == "1":
                result, verifier = verify_text(text, use_enhanced=True)
                if not result.is_consistent:
                    repair_choice = input("\nWould you like to repair the inconsistencies? (y/n): ").lower()
                    if repair_choice == 'y':
                        repair_text(text, verifier)
            
            elif choice == "2":
                result, verifier = verify_text(text, use_enhanced=False)
                if not result.is_consistent:
                    repair_choice = input("\nWould you like to repair the inconsistencies? (y/n): ").lower()
                    if repair_choice == 'y':
                        repair_text(text, verifier)
            
            elif choice == "3":
                standard_result, enhanced_result = compare_verifiers(text)
                if not enhanced_result.is_consistent:
                    repair_choice = input("\nWould you like to repair the inconsistencies? (y/n): ").lower()
                    if repair_choice == 'y':
                        verifier = EnhancedConsistencyVerifier()
                        repair_text(text, verifier)
        
        elif choice == "5":
            print_info("Thank you for using the LLM Consistency Verifier!")
            break
        
        else:
            print_error("Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main() 