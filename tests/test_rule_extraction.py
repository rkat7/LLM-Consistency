#!/usr/bin/env python3
import os
import sys
import json
import time
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_consistency_verifier.utils.rule_extractor import RuleExtractor, SpacyRuleExtractor
from llm_consistency_verifier.utils.llm_interface import LLMInterface
from llm_consistency_verifier.config.config import Config

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def compare_extraction_methods(test_cases, output_dir="results"):
    """
    Compare rule extraction methods: pattern-based, spaCy-based, and LLM-based.
    
    Args:
        test_cases: List of test cases
        output_dir: Directory to save results
    """
    # Initialize extractors
    basic_extractor = RuleExtractor()
    
    try:
        spacy_extractor = SpacyRuleExtractor()
        has_spacy = True
    except Exception:
        logger.warning("Failed to initialize SpacyRuleExtractor, skipping spaCy comparison")
        has_spacy = False
    
    llm_interface = LLMInterface()
    
    results = {
        "pattern": {"times": [], "rule_counts": [], "rules_by_case": []},
        "spacy": {"times": [], "rule_counts": [], "rules_by_case": []} if has_spacy else None,
        "llm": {"times": [], "rule_counts": [], "rules_by_case": []}
    }
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process test cases
    for i, case in enumerate(test_cases):
        text = case["text"]
        logger.info(f"Processing test case {i+1}/{len(test_cases)}")
        
        # 1. Pattern-based extraction
        start_time = time.time()
        pattern_rules = basic_extractor.extract_rules(text)
        pattern_time = time.time() - start_time
        
        results["pattern"]["times"].append(pattern_time)
        results["pattern"]["rule_counts"].append(len(pattern_rules))
        results["pattern"]["rules_by_case"].append(pattern_rules)
        
        # 2. spaCy-based extraction
        if has_spacy:
            start_time = time.time()
            spacy_rules = spacy_extractor.extract_rules(text)
            spacy_time = time.time() - start_time
            
            results["spacy"]["times"].append(spacy_time)
            results["spacy"]["rule_counts"].append(len(spacy_rules))
            results["spacy"]["rules_by_case"].append(spacy_rules)
        
        # 3. LLM-based extraction
        start_time = time.time()
        llm_rules = llm_interface.extract_logical_rules(text)
        llm_time = time.time() - start_time
        
        results["llm"]["times"].append(llm_time)
        results["llm"]["rule_counts"].append(len(llm_rules))
        results["llm"]["rules_by_case"].append(llm_rules)
    
    # Calculate summary statistics
    for method in results:
        if results[method] is None:
            continue
            
        total_rules = sum(results[method]["rule_counts"])
        total_time = sum(results[method]["times"])
        
        results[method]["total_rules"] = total_rules
        results[method]["total_time"] = total_time
        results[method]["avg_rules_per_case"] = total_rules / len(test_cases) if test_cases else 0
        results[method]["avg_time_per_case"] = total_time / len(test_cases) if test_cases else 0
    
    # Save results
    with open(os.path.join(output_dir, "extraction_comparison.json"), "w") as f:
        # Convert rules to string representations for JSON serialization
        serializable_results = {
            method: {
                k: (
                    v if k not in ["rules_by_case"] else 
                    [[str(rule) for rule in case_rules] for case_rules in v]
                ) 
                for k, v in data.items()
            } if data is not None else None
            for method, data in results.items()
        }
        json.dump(serializable_results, f, indent=2)
    
    # Print summary
    print("\n=== Rule Extraction Comparison ===")
    print(f"Test cases: {len(test_cases)}")
    
    for method in ["pattern", "spacy", "llm"]:
        if method == "spacy" and not has_spacy:
            continue
            
        print(f"\n{method.upper()}-based extraction:")
        print(f"  Total rules extracted: {results[method]['total_rules']}")
        print(f"  Avg rules per case: {results[method]['avg_rules_per_case']:.2f}")
        print(f"  Total time: {results[method]['total_time']:.2f}s")
        print(f"  Avg time per case: {results[method]['avg_time_per_case']:.2f}s")
    
    return results

def main():
    """Run rule extraction comparison test."""
    test_cases = [
        {
            "text": "All birds can fly. Penguins are birds. Penguins cannot fly.",
            "is_consistent": False,
            "category": "direct_contradiction"
        },
        {
            "text": "If a student studies, they will pass the exam. If they pass the exam, they will graduate. Alice studied but did not graduate.",
            "is_consistent": False,
            "category": "complex_implication"
        },
        {
            "text": "Every integer is either even or odd. There exists an integer that is both even and odd.",
            "is_consistent": False,
            "category": "quantifier_reasoning"
        },
        {
            "text": "All roses are flowers. Some flowers are red. Therefore, some roses are red.",
            "is_consistent": True,
            "category": "consistent"
        },
        {
            "text": "If it rains, the ground gets wet. The ground is wet. Therefore, it rained.",
            "is_consistent": False,
            "category": "logical_fallacy"
        }
    ]
    
    compare_extraction_methods(test_cases)

if __name__ == "__main__":
    main() 