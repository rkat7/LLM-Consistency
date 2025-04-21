#!/usr/bin/env python3
import sys
import time
import logging
import argparse
from typing import List, Dict, Any

from llm_consistency_verifier.core.consistency_verifier import ConsistencyVerifier
from llm_consistency_verifier.core.verification_engine import VerificationResult
from llm_consistency_verifier.utils.rule_extractor import SpacyRuleExtractor
from llm_consistency_verifier.config.config import Config

# Configure logging - change level to WARNING to reduce terminal output
logging.basicConfig(level=logging.WARNING)

def print_workflow_diagram():
    """Print a visual diagram of the workflow steps."""
    diagram = """
╔════════════════════════════════ WORKFLOW DIAGRAM ═══════════════════════════════╗
║                                                                                 ║
║  ┌─────────────┐     ┌────────────────────┐     ┌────────────────────┐         ║
║  │  Input Text │────►│ Rule Extraction    │────►│ Formalization      │         ║
║  │             │     │ [Pattern Analyzer] │     │ [Logical Converter]│         ║
║  └─────────────┘     └────────────────────┘     └────────────────────┘         ║
║          │                                                 │                    ║
║          │                                                 ▼                    ║
║          │                                       ┌────────────────────┐         ║
║          │                                       │ Verification       │         ║
║          │                                       │ [Z3 Solver]        │         ║
║          │                                       └────────────────────┘         ║
║          │                                                 │                    ║
║          │                                                 ▼                    ║
║          │                                       ┌────────────────────┐         ║
║          │                                       │ Analysis Results   │         ║
║          │                                       │ [Consistent/       │         ║
║          │                                       │  Inconsistent]     │         ║
║          │                                       └────────────────────┘         ║
║          │                                                 │                    ║
║          │               If inconsistent                   ▼                    ║
║          └───────────────────────────────────► ┌────────────────────┐         ║
║                                                │ Repair Suggestion  │         ║
║                                                │ [LLM]             │         ║
║                                                └────────────────────┘         ║
║                                                                                 ║
╚═════════════════════════════════════════════════════════════════════════════════╝
"""
    print(diagram)
    time.sleep(1)  # Pause to let the user see the diagram

def print_step(step_num, title, content=None, delay=0.5):
    """Print a workflow step with a consistent format."""
    print(f"\n{'='*80}")
    print(f"STEP {step_num}: {title}")
    print(f"{'='*80}")
    if content:
        print(content)
    time.sleep(delay)  # Add a small delay for visual effect

def print_arrow():
    """Print a downward arrow to visualize flow."""
    print("\n        ↓ ↓ ↓\n")
    time.sleep(0.3)

def print_workflow_visualization(text: str, verbose: bool = False):
    """
    Visualize the entire workflow of the consistency verification system.
    
    Args:
        text: The input text to verify
        verbose: Whether to print verbose details
    """
    # Display the workflow diagram
    print_workflow_diagram()
    
    # Initialize the verification components
    verifier = ConsistencyVerifier()
    rule_extractor = SpacyRuleExtractor()
    
    # STEP 1: Input Text
    print_step(1, "INPUT TEXT", text)
    print_arrow()
    
    # STEP 2: Rule Extraction
    start_time = time.time()
    print_step(2, "RULE EXTRACTION [Pattern-Based Analyzer]", "Extracting logical rules from natural language...")
    rules = rule_extractor.extract_rules(text)
    
    # Display the extracted rules
    rules_text = "\n".join([f"• {rule.get('type', 'Unknown').upper()}: {rule.get('original_text', '')}" 
                           for rule in rules])
    print(f"\nExtracted {len(rules)} logical rules:\n{rules_text}")
    print(f"\nRule extraction completed in {time.time() - start_time:.2f}s")
    print_arrow()
    
    # STEP 3: Formalization
    print_step(3, "FORMALIZATION [Logical Conversion]", "Converting natural language rules into formal logic...")
    statements = text.split('.')
    statements = [s.strip() + '.' for s in statements if s.strip()]
    
    normalized_rules = []
    for rule in rules:
        rule_type = rule.get('type', 'unknown')
        if rule_type == 'assertion':
            normalized_rules.append(f"ASSERTION: {rule.get('statement', '')}")
        elif rule_type == 'universal':
            normalized_rules.append(f"UNIVERSAL: ∀x: {rule.get('subject', '')}(x) → {rule.get('predicate', '')}(x)")
        elif rule_type == 'negation':
            normalized_rules.append(f"NEGATION: ¬({rule.get('statement', '')})")
    
    for rule in normalized_rules:
        print(f"• {rule}")
    print_arrow()
    
    # STEP 4: Verification 
    print_step(4, "VERIFICATION [Z3 Solver]", "Checking logical consistency...")
    start_time = time.time()
    result = verifier.verify(text)
    print(f"Verification completed in {time.time() - start_time:.2f}s")
    print_arrow()
    
    # STEP 5: Results Analysis
    print_step(5, "ANALYSIS RESULTS", "Analyzing verification results...")
    
    if result.is_consistent:
        print("\n✅ TEXT IS LOGICALLY CONSISTENT")
        print("No contradictions or inconsistencies were found in the provided statements.")
    else:
        print("\n❌ TEXT IS LOGICALLY INCONSISTENT")
        print("The following inconsistencies were detected:")
        
        for i, inconsistency in enumerate(result.inconsistencies, 1):
            print(f"{i}. {inconsistency}")
            
        if result.proof:
            print(f"\nProof: {result.proof}")
    print_arrow()
    
    # STEP 6: Repair (if needed)
    if not result.is_consistent:
        print_step(6, "REPAIR SUGGESTION [LLM]", "Generating possible fixes...")
        
        # The repair method returns a string, not an object with repaired_text attribute
        repaired_text = verifier.repair(text)
        
        print("\nSuggested repair:")
        print(f"\n{repaired_text}")
        
        # Verify the repair worked
        print("\nVerifying repair...")
        repair_verification = verifier.verify(repaired_text)
        
        if repair_verification.is_consistent:
            print("\n✅ REPAIR SUCCESSFUL")
        else:
            print("\n❌ REPAIR FAILED")
            print("The suggested repair still contains inconsistencies.")

def main():
    parser = argparse.ArgumentParser(description='Visualize the consistency verification workflow')
    parser.add_argument('text', help='Text to verify for logical consistency')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show verbose output')
    
    args = parser.parse_args()
    print_workflow_visualization(args.text, args.verbose)

if __name__ == "__main__":
    main() 