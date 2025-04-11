#!/usr/bin/env python3
import os
import sys
import time
import json
import logging
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np
import re

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_consistency_verifier import ConsistencyVerifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BaselineChecker:
    """A simple baseline that checks for logical inconsistencies."""
    
    def check_consistency(self, text: str) -> Dict[str, Any]:
        """Check text for logical inconsistencies using simple heuristics.
        
        Args:
            text: The text to check
            
        Returns:
            Dictionary with consistency results
        """
        contradictions = []
        sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
        
        # Check for direct negations
        for i, s1 in enumerate(sentences):
            for s2 in sentences[i+1:]:
                s1_lower = s1.lower()
                s2_lower = s2.lower()
                
                # Check for statements that might contradict each other
                if self._check_direct_contradiction(s1_lower, s2_lower):
                    contradictions.append(f"Potential contradiction between '{s1}' and '{s2}'")
                
                # Check for rule violations (e.g., "All X are Y" but "Z is X and not Y")
                if self._check_rule_violation(s1_lower, s2_lower, sentences):
                    contradictions.append(f"Potential rule violation between '{s1}' and '{s2}'")
        
        is_consistent = len(contradictions) == 0
        return {
            "is_consistent": is_consistent,
            "inconsistencies": contradictions if not is_consistent else []
        }
    
    def _check_direct_contradiction(self, s1: str, s2: str) -> bool:
        """Check if two statements directly contradict each other."""
        # Check for presence of negation in one but not the other
        contains_not_s1 = "not " in s1 or " no " in s1 or "n't " in s1
        contains_not_s2 = "not " in s2 or " no " in s2 or "n't " in s2
        
        if contains_not_s1 != contains_not_s2:
            # Remove negation words for comparison
            clean_s1 = s1.replace("not ", "").replace(" no ", " ").replace("n't ", " ")
            clean_s2 = s2.replace("not ", "").replace(" no ", " ").replace("n't ", " ")
            
            # Calculate word overlap
            words_s1 = set(clean_s1.split())
            words_s2 = set(clean_s2.split())
            common_words = words_s1.intersection(words_s2)
            
            # If they share enough words, they might be contradicting
            if len(common_words) >= min(3, len(words_s1) // 2):
                return True
        
        return False
    
    def _check_rule_violation(self, s1: str, s2: str, all_sentences: List[str]) -> bool:
        """Check if statements violate logical rules."""
        # Check for universal statements and exceptions
        if s1.startswith("all ") or s1.startswith("every "):
            # Extract the rule: "All X are Y"
            match = re.search(r"all (.+?) (are|is|have|has) (.+)", s1)
            if match:
                subject = match.group(1).strip()
                verb = match.group(2).strip()
                predicate = match.group(3).strip()
                
                # Look for statements about the subject that contradict the predicate
                if subject in s2:
                    opposite_predicate = f"not {predicate}"
                    if opposite_predicate in s2 or (predicate in s2 and ("not " in s2 or "n't" in s2)):
                        return True
        
        # Check for implications
        if "if " in s1 and " then " in s1:
            # Extract rule: "If X then Y"
            match = re.search(r"if (.+?) then (.+)", s1)
            if match:
                condition = match.group(1).strip()
                result = match.group(2).strip()
                
                # Check for statements that affirm condition but deny result
                condition_true = any(condition in s and "not" not in s for s in all_sentences)
                result_false = any(result in s and ("not" in s or "n't" in s) for s in all_sentences)
                
                if condition_true and result_false:
                    return True
        
        return False

class BaselineComparison:
    """Compares the verifier against a baseline method."""
    
    def __init__(self, test_cases_path: str = None):
        """Initialize with test cases.
        
        Args:
            test_cases_path: Path to test cases JSON file (optional)
        """
        self.verifier = ConsistencyVerifier()
        self.baseline = BaselineChecker()
        self.test_cases = []
        
        if test_cases_path:
            try:
                with open(test_cases_path, 'r') as f:
                    data = json.load(f)
                    self.test_cases = data.get('test_cases', [])
                logger.info(f"Loaded {len(self.test_cases)} test cases from {test_cases_path}")
            except Exception as e:
                logger.error(f"Failed to load test cases: {str(e)}")
                raise
    
    def run_comparison(self) -> Dict[str, Any]:
        """Run comparison between verifier and baseline.
        
        Returns:
            Dictionary with comparison results
        """
        results = {
            "verifier": {
                "correct": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "precision": 0,
                "recall": 0,
                "f1": 0,
                "accuracy": 0,
                "avg_time": 0
            },
            "baseline": {
                "correct": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "precision": 0,
                "recall": 0,
                "f1": 0,
                "accuracy": 0,
                "avg_time": 0
            },
            "by_category": {},
            "case_results": []
        }
        
        verifier_time_total = 0
        baseline_time_total = 0
        
        # Initialize category metrics
        categories = set(case.get("category", "uncategorized") for case in self.test_cases)
        for category in categories:
            results["by_category"][category] = {
                "count": 0,
                "verifier_correct": 0,
                "baseline_correct": 0
            }
        
        # Process each test case
        for case in self.test_cases:
            category = case.get("category", "uncategorized")
            ground_truth = not case.get("is_consistent", True)  # True if inconsistent
            
            case_result = {
                "text": case["text"],
                "ground_truth": ground_truth,
                "verifier_result": None,
                "baseline_result": None,
                "verifier_time": 0,
                "baseline_time": 0
            }
            
            # Run verifier
            start_time = time.time()
            verifier_result = self.verifier.verify(case["text"])
            verifier_time = time.time() - start_time
            
            verifier_prediction = not verifier_result.is_consistent  # True if inconsistent
            case_result["verifier_result"] = verifier_prediction
            case_result["verifier_time"] = verifier_time
            case_result["verifier_inconsistencies"] = verifier_result.inconsistencies
            
            verifier_time_total += verifier_time
            
            # Run baseline
            start_time = time.time()
            baseline_result = self.baseline.check_consistency(case["text"])
            baseline_time = time.time() - start_time
            
            baseline_prediction = not baseline_result["is_consistent"]  # True if inconsistent
            case_result["baseline_result"] = baseline_prediction
            case_result["baseline_time"] = baseline_time
            case_result["baseline_inconsistencies"] = baseline_result["inconsistencies"]
            
            baseline_time_total += baseline_time
            
            # Update metrics
            results["by_category"][category]["count"] += 1
            
            if verifier_prediction == ground_truth:
                results["verifier"]["correct"] += 1
                results["by_category"][category]["verifier_correct"] += 1
            elif verifier_prediction and not ground_truth:
                results["verifier"]["false_positives"] += 1
            elif not verifier_prediction and ground_truth:
                results["verifier"]["false_negatives"] += 1
            
            if baseline_prediction == ground_truth:
                results["baseline"]["correct"] += 1
                results["by_category"][category]["baseline_correct"] += 1
            elif baseline_prediction and not ground_truth:
                results["baseline"]["false_positives"] += 1
            elif not baseline_prediction and ground_truth:
                results["baseline"]["false_negatives"] += 1
            
            results["case_results"].append(case_result)
        
        # Calculate overall metrics
        total_cases = len(self.test_cases)
        
        if total_cases > 0:
            results["verifier"]["accuracy"] = results["verifier"]["correct"] / total_cases
            results["baseline"]["accuracy"] = results["baseline"]["correct"] / total_cases
            
            results["verifier"]["avg_time"] = verifier_time_total / total_cases
            results["baseline"]["avg_time"] = baseline_time_total / total_cases
        
        # Calculate precision, recall, F1
        for method in ["verifier", "baseline"]:
            true_positives = results[method]["correct"] - results[method]["false_negatives"]
            
            if true_positives + results[method]["false_positives"] > 0:
                results[method]["precision"] = true_positives / (true_positives + results[method]["false_positives"])
            
            inconsistent_cases = sum(1 for case in self.test_cases if not case.get("is_consistent", True))
            if inconsistent_cases > 0:
                results[method]["recall"] = true_positives / inconsistent_cases
            
            if results[method]["precision"] + results[method]["recall"] > 0:
                results[method]["f1"] = 2 * (results[method]["precision"] * results[method]["recall"]) / (results[method]["precision"] + results[method]["recall"])
        
        # Calculate category metrics
        for category in results["by_category"]:
            category_count = results["by_category"][category]["count"]
            if category_count > 0:
                results["by_category"][category]["verifier_accuracy"] = results["by_category"][category]["verifier_correct"] / category_count
                results["by_category"][category]["baseline_accuracy"] = results["by_category"][category]["baseline_correct"] / category_count
        
        return results
    
    def visualize_results(self, results: Dict[str, Any], output_dir: str = "results"):
        """Generate visualizations of comparison results.
        
        Args:
            results: The results dictionary
            output_dir: Directory to save visualizations
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Plot 1: Accuracy comparison
        self._plot_accuracy_comparison(results, output_dir)
        
        # Plot 2: Precision, recall, F1 comparison
        self._plot_prf_comparison(results, output_dir)
        
        # Plot 3: Category accuracy comparison
        self._plot_category_comparison(results, output_dir)
        
        # Plot 4: Processing time comparison
        self._plot_time_comparison(results, output_dir)
        
        # Save results as JSON
        with open(os.path.join(output_dir, "baseline_comparison.json"), 'w') as f:
            # Filter out non-serializable objects
            serializable_results = {
                k: v for k, v in results.items() 
                if k != "case_results"
            }
            serializable_results["case_results"] = []
            for case in results["case_results"]:
                serializable_case = {
                    k: (v if not isinstance(v, list) else [str(i) for i in v]) 
                    for k, v in case.items()
                }
                serializable_results["case_results"].append(serializable_case)
            
            json.dump(serializable_results, f, indent=2)
    
    def _plot_accuracy_comparison(self, results: Dict[str, Any], output_dir: str):
        """Plot accuracy comparison."""
        methods = ["Baseline", "Neural-Symbolic Verifier"]
        accuracy = [results["baseline"]["accuracy"], results["verifier"]["accuracy"]]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(methods, accuracy, color=['lightgray', 'green'])
        ax.set_ylabel('Accuracy')
        ax.set_title('Inconsistency Detection Accuracy')
        ax.set_ylim(0, 1)
        
        for i, v in enumerate(accuracy):
            ax.text(i, v + 0.02, f'{v:.2f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"))
        plt.close()
    
    def _plot_prf_comparison(self, results: Dict[str, Any], output_dir: str):
        """Plot precision, recall, F1 comparison."""
        methods = ["Baseline", "Neural-Symbolic Verifier"]
        metrics = ["Precision", "Recall", "F1 Score"]
        
        baseline_values = [
            results["baseline"]["precision"],
            results["baseline"]["recall"],
            results["baseline"]["f1"]
        ]
        
        verifier_values = [
            results["verifier"]["precision"],
            results["verifier"]["recall"],
            results["verifier"]["f1"]
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, baseline_values, width, label='Baseline')
        ax.bar(x + width/2, verifier_values, width, label='Neural-Symbolic Verifier')
        
        ax.set_ylabel('Score')
        ax.set_title('Precision, Recall, and F1 Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.set_ylim(0, 1)
        
        for i, v in enumerate(baseline_values):
            ax.text(i - width/2, v + 0.02, f'{v:.2f}', ha='center')
        
        for i, v in enumerate(verifier_values):
            ax.text(i + width/2, v + 0.02, f'{v:.2f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "prf_comparison.png"))
        plt.close()
    
    def _plot_category_comparison(self, results: Dict[str, Any], output_dir: str):
        """Plot category accuracy comparison."""
        categories = list(results["by_category"].keys())
        baseline_accuracy = [results["by_category"][c]["baseline_accuracy"] for c in categories]
        verifier_accuracy = [results["by_category"][c]["verifier_accuracy"] for c in categories]
        
        # Format category names for display
        display_categories = [c.replace('_', ' ').title() for c in categories]
        
        x = np.arange(len(categories))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, baseline_accuracy, width, label='Baseline')
        ax.bar(x + width/2, verifier_accuracy, width, label='Neural-Symbolic Verifier')
        
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy by Category')
        ax.set_xticks(x)
        ax.set_xticklabels(display_categories)
        ax.legend()
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "category_comparison.png"))
        plt.close()
    
    def _plot_time_comparison(self, results: Dict[str, Any], output_dir: str):
        """Plot processing time comparison."""
        methods = ["Baseline", "Neural-Symbolic Verifier"]
        times = [results["baseline"]["avg_time"], results["verifier"]["avg_time"]]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(methods, times, color=['lightgray', 'green'])
        ax.set_ylabel('Average Processing Time (seconds)')
        ax.set_title('Processing Time Comparison')
        
        for i, v in enumerate(times):
            ax.text(i, v + 0.1, f'{v:.2f}s', ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "time_comparison.png"))
        plt.close()

def main():
    """Run baseline comparison and visualization."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare ConsistencyVerifier to baseline method')
    parser.add_argument('--test-cases', '-t', help='Path to test cases JSON file')
    parser.add_argument('--output-dir', '-o', default='results', help='Directory to save results')
    
    args = parser.parse_args()
    
    logger.info("Starting baseline comparison")
    
    comparison = BaselineComparison(args.test_cases)
    results = comparison.run_comparison()
    
    logger.info("Baseline comparison completed")
    logger.info(f"Verifier accuracy: {results['verifier']['accuracy']:.2f}")
    logger.info(f"Baseline accuracy: {results['baseline']['accuracy']:.2f}")
    
    comparison.visualize_results(results, args.output_dir)
    logger.info(f"Results saved to {args.output_dir} directory")

if __name__ == "__main__":
    main() 