#!/usr/bin/env python3
import os
import sys
import time
import json
import logging
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np

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

class BenchmarkSuite:
    """Benchmark suite for the consistency verifier."""
    
    def __init__(self, benchmark_path: str = None):
        """Initialize the benchmark suite.
        
        Args:
            benchmark_path: Path to benchmark JSON file (optional)
        """
        self.verifier = ConsistencyVerifier()
        self.benchmarks = {}
        
        if benchmark_path:
            try:
                with open(benchmark_path, 'r') as f:
                    data = json.load(f)
                    test_cases = data.get('test_cases', [])
                    
                # Group test cases by category
                for case in test_cases:
                    category = case.get('category', 'uncategorized')
                    if category not in self.benchmarks:
                        self.benchmarks[category] = []
                    self.benchmarks[category].append(case['text'])
                    
                logger.info(f"Loaded {len(test_cases)} test cases across {len(self.benchmarks)} categories")
            except Exception as e:
                logger.error(f"Failed to load benchmark cases: {str(e)}")
                raise
        else:
            # Default benchmark categories and test cases
            self.benchmarks = {
                "direct_contradiction": [
                    "A is true. A is false.",
                    "All birds can fly. Penguins are birds. Penguins cannot fly.",
                    "John is tall. John is short."
                ],
                "complex_implication": [
                    "If A then B. If B then C. A is true. C is false.",
                    "If the policy is implemented, taxes will increase. If taxes increase, spending will decrease. The policy is implemented. Spending has increased.",
                    "If someone is a doctor, they have a medical degree. If someone has a medical degree, they went to medical school. Dr. Smith is a doctor who didn't go to medical school."
                ],
                "quantifier_reasoning": [
                    "All humans are mortal. Socrates is human. Socrates is immortal.",
                    "Every integer is either even or odd. There exists an integer that is both even and odd.",
                    "All students have taken at least one exam. No failed student has passed any exam. Some students have failed."
                ],
                "real_world_examples": [
                    "If a student studies, they will pass the exam. If they pass the exam, they will graduate. Alice studied but did not graduate.",
                    "All politicians are dishonest. Senator Jones is honest. Senator Jones is a politician.",
                    "Using fossil fuels increases carbon emissions. Increased carbon emissions lead to climate change. Alternative energy doesn't use fossil fuels. Alternative energy will not affect climate change."
                ]
            }
    
    def run_benchmark(self) -> Dict[str, Any]:
        """Run all benchmarks and collect results.
        
        Returns:
            Dictionary of results by category
        """
        results = {}
        overall_results = {
            "total_cases": 0,
            "detection_count": 0,
            "repair_success_count": 0,
            "avg_verification_time": 0,
            "avg_repair_time": 0,
            "avg_inconsistencies": 0
        }
        
        for category, cases in self.benchmarks.items():
            logger.info(f"Running benchmark for category: {category}")
            
            category_results = {
                "detection_rate": 0,
                "repair_success_rate": 0,
                "avg_inconsistencies": 0,
                "avg_verification_time": 0,
                "avg_repair_time": 0,
                "examples": []
            }
            
            detection_count = 0
            repair_success_count = 0
            total_inconsistencies = 0
            total_verification_time = 0
            total_repair_time = 0
            
            for i, case in enumerate(cases):
                example_result = {
                    "text": case,
                    "verified": False,
                    "is_consistent": None,
                    "inconsistencies": [],
                    "verification_time": 0,
                    "repaired": False,
                    "repaired_text": "",
                    "repair_time": 0
                }
                
                # Verify the case
                start_time = time.time()
                try:
                    result = self.verifier.verify(case)
                    verification_time = time.time() - start_time
                    
                    example_result["verified"] = True
                    example_result["is_consistent"] = result.is_consistent
                    example_result["inconsistencies"] = result.inconsistencies
                    example_result["verification_time"] = verification_time
                    
                    total_verification_time += verification_time
                    
                    # All benchmark cases should be inconsistent by design
                    # If we find an inconsistency, count it as a success
                    if not result.is_consistent:
                        detection_count += 1
                        total_inconsistencies += len(result.inconsistencies)
                        
                        # Try repair
                        start_time = time.time()
                        repaired = self.verifier.repair(case)
                        repair_time = time.time() - start_time
                        
                        example_result["repaired"] = True
                        example_result["repaired_text"] = repaired
                        example_result["repair_time"] = repair_time
                        
                        total_repair_time += repair_time
                        
                        # Check if repair fixed the inconsistencies
                        repair_result = self.verifier.verify(repaired)
                        if repair_result.is_consistent:
                            repair_success_count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing case {i} in {category}: {str(e)}")
                    example_result["error"] = str(e)
                
                category_results["examples"].append(example_result)
            
            # Calculate category metrics
            total_cases = len(cases)
            category_results["detection_rate"] = detection_count / total_cases if total_cases > 0 else 0
            category_results["repair_success_rate"] = repair_success_count / detection_count if detection_count > 0 else 0
            category_results["avg_inconsistencies"] = total_inconsistencies / detection_count if detection_count > 0 else 0
            category_results["avg_verification_time"] = total_verification_time / total_cases if total_cases > 0 else 0
            category_results["avg_repair_time"] = total_repair_time / detection_count if detection_count > 0 else 0
            
            results[category] = category_results
            
            # Update overall metrics
            overall_results["total_cases"] += total_cases
            overall_results["detection_count"] += detection_count
            overall_results["repair_success_count"] += repair_success_count
            overall_results["avg_verification_time"] += total_verification_time
            overall_results["avg_repair_time"] += total_repair_time
            overall_results["avg_inconsistencies"] += total_inconsistencies
            
            logger.info(f"Completed benchmark for category: {category}")
        
        # Calculate overall averages
        if overall_results["total_cases"] > 0:
            overall_results["avg_verification_time"] /= overall_results["total_cases"]
            overall_results["detection_rate"] = overall_results["detection_count"] / overall_results["total_cases"]
        
        if overall_results["detection_count"] > 0:
            overall_results["avg_repair_time"] /= overall_results["detection_count"]
            overall_results["repair_success_rate"] = overall_results["repair_success_count"] / overall_results["detection_count"]
            overall_results["avg_inconsistencies"] /= overall_results["detection_count"]
        
        results["overall"] = overall_results
        return results
    
    def visualize_results(self, results: Dict[str, Any], output_dir: str = "results"):
        """Generate visualizations of benchmark results.
        
        Args:
            results: The results dictionary
            output_dir: Directory to save visualizations
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Extract category results (excluding overall)
        categories = [cat for cat in results.keys() if cat != "overall"]
        
        # Plot 1: Detection and repair rates by category
        self._plot_rates_by_category(results, categories, output_dir)
        
        # Plot 2: Average inconsistencies by category
        self._plot_inconsistencies(results, categories, output_dir)
        
        # Plot 3: Processing times by category
        self._plot_times(results, categories, output_dir)
        
        # Save results as JSON
        with open(os.path.join(output_dir, "benchmark_results.json"), 'w') as f:
            json.dump(results, f, indent=2)
    
    def _plot_rates_by_category(self, results: Dict[str, Any], categories: List[str], output_dir: str):
        """Plot detection and repair rates by category."""
        detection_rates = [results[cat]["detection_rate"] for cat in categories]
        repair_rates = [results[cat]["repair_success_rate"] for cat in categories]
        
        # Format category names for display
        display_categories = [cat.replace('_', ' ').title() for cat in categories]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(categories))
        width = 0.35
        
        ax.bar(x - width/2, detection_rates, width, label='Detection Rate')
        ax.bar(x + width/2, repair_rates, width, label='Repair Success Rate')
        
        ax.set_ylabel('Rate')
        ax.set_title('Detection and Repair Rates by Category')
        ax.set_xticks(x)
        ax.set_xticklabels(display_categories)
        ax.legend()
        ax.set_ylim(0, 1.1)
        
        for i, value in enumerate(detection_rates):
            ax.text(i - width/2, value + 0.05, f'{value:.2f}', ha='center')
        
        for i, value in enumerate(repair_rates):
            ax.text(i + width/2, value + 0.05, f'{value:.2f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "benchmark_rates.png"))
        plt.close()
    
    def _plot_inconsistencies(self, results: Dict[str, Any], categories: List[str], output_dir: str):
        """Plot average inconsistencies by category."""
        inconsistencies = [results[cat]["avg_inconsistencies"] for cat in categories]
        
        # Format category names for display
        display_categories = [cat.replace('_', ' ').title() for cat in categories]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.bar(display_categories, inconsistencies, color='orange')
        
        ax.set_ylabel('Average Number of Inconsistencies')
        ax.set_title('Average Inconsistencies Detected by Category')
        ax.set_ylim(bottom=0)
        
        for i, value in enumerate(inconsistencies):
            ax.text(i, value + 0.1, f'{value:.2f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "benchmark_inconsistencies.png"))
        plt.close()
    
    def _plot_times(self, results: Dict[str, Any], categories: List[str], output_dir: str):
        """Plot processing times by category."""
        verification_times = [results[cat]["avg_verification_time"] for cat in categories]
        repair_times = [results[cat]["avg_repair_time"] for cat in categories]
        
        # Format category names for display
        display_categories = [cat.replace('_', ' ').title() for cat in categories]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(categories))
        width = 0.35
        
        ax.bar(x - width/2, verification_times, width, label='Avg. Verification Time (s)')
        ax.bar(x + width/2, repair_times, width, label='Avg. Repair Time (s)')
        
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Average Processing Time by Category')
        ax.set_xticks(x)
        ax.set_xticklabels(display_categories)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "benchmark_times.png"))
        plt.close()

def main():
    """Run benchmark suite and visualization."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run benchmark suite for the ConsistencyVerifier')
    parser.add_argument('--benchmark-file', '-b', help='Path to benchmark JSON file')
    parser.add_argument('--output-dir', '-o', default='results', help='Directory to save results')
    
    args = parser.parse_args()
    
    logger.info("Starting benchmark suite")
    
    suite = BenchmarkSuite(args.benchmark_file)
    results = suite.run_benchmark()
    
    logger.info("Benchmark suite completed")
    logger.info(f"Overall detection rate: {results['overall']['detection_rate']:.2f}")
    logger.info(f"Overall repair success rate: {results['overall']['repair_success_rate']:.2f}")
    
    suite.visualize_results(results, args.output_dir)
    logger.info(f"Results saved to {args.output_dir} directory")

if __name__ == "__main__":
    main() 