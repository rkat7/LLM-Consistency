#!/usr/bin/env python3
import time
import statistics
import logging
import os
import sys
import json
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_consistency_verifier import ConsistencyVerifier
from llm_consistency_verifier.config.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """Measures performance metrics for the consistency verifier."""
    
    def __init__(self, test_cases_path: str = None):
        """Initialize with test cases.
        
        Args:
            test_cases_path: Path to test cases JSON file (optional)
        """
        self.verifier = ConsistencyVerifier()
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
    
    def measure_performance(self, iterations: int = 3) -> Dict[str, Any]:
        """Measure performance metrics across all test cases.
        
        Args:
            iterations: Number of iterations to run for each test case
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            "verification_times": [],
            "repair_times": [],
            "success_rate": 0,
            "inconsistency_detection_rate": 0,
            "repair_success_rate": 0,
            "by_category": {}
        }
        
        total_tests = len(self.test_cases) * iterations
        success_count = 0
        detection_count = 0
        repair_success_count = 0
        
        for case in self.test_cases:
            category = case.get("category", "uncategorized")
            
            # Initialize category metrics if needed
            if category not in metrics["by_category"]:
                metrics["by_category"][category] = {
                    "verification_times": [],
                    "repair_times": [],
                    "success_rate": 0,
                    "detection_rate": 0,
                    "repair_success_rate": 0,
                    "count": 0
                }
            
            case_success = 0
            case_detection = 0
            case_repair_success = 0
            
            for _ in range(iterations):
                metrics["by_category"][category]["count"] += 1
                
                # Measure verification time
                start = time.time()
                verification_result = self.verifier.verify(case["text"])
                verification_time = time.time() - start
                
                metrics["verification_times"].append(verification_time)
                metrics["by_category"][category]["verification_times"].append(verification_time)
                
                # Check if verification matches ground truth
                if verification_result.is_consistent == case.get("is_consistent", True):
                    success_count += 1
                    case_success += 1
                
                # If inconsistent and expected to be, count as successful detection
                if not verification_result.is_consistent and not case.get("is_consistent", True):
                    detection_count += 1
                    case_detection += 1
                    
                    # Try repair if inconsistent
                    start = time.time()
                    repaired = self.verifier.repair(case["text"])
                    repair_time = time.time() - start
                    
                    metrics["repair_times"].append(repair_time)
                    metrics["by_category"][category]["repair_times"].append(repair_time)
                    
                    # Check if repair fixed inconsistencies
                    repair_result = self.verifier.verify(repaired)
                    if repair_result.is_consistent:
                        repair_success_count += 1
                        case_repair_success += 1
            
            # Calculate category-specific rates
            category_iterations = iterations
            if category_iterations > 0:
                metrics["by_category"][category]["success_rate"] = case_success / category_iterations
                metrics["by_category"][category]["detection_rate"] = case_detection / category_iterations
                if case_detection > 0:
                    metrics["by_category"][category]["repair_success_rate"] = case_repair_success / case_detection
                else:
                    metrics["by_category"][category]["repair_success_rate"] = 0
        
        # Calculate overall metrics
        if total_tests > 0:
            metrics["success_rate"] = success_count / total_tests
            metrics["inconsistency_detection_rate"] = detection_count / total_tests
        
        if detection_count > 0:
            metrics["repair_success_rate"] = repair_success_count / detection_count
        
        # Calculate timing statistics
        if metrics["verification_times"]:
            metrics["avg_verification_time"] = statistics.mean(metrics["verification_times"])
            if len(metrics["verification_times"]) > 1:
                metrics["std_verification_time"] = statistics.stdev(metrics["verification_times"])
            else:
                metrics["std_verification_time"] = 0
        
        if metrics["repair_times"]:
            metrics["avg_repair_time"] = statistics.mean(metrics["repair_times"])
            if len(metrics["repair_times"]) > 1:
                metrics["std_repair_time"] = statistics.stdev(metrics["repair_times"])
            else:
                metrics["std_repair_time"] = 0
        
        return metrics
    
    def visualize_results(self, metrics: Dict[str, Any], output_dir: str = "results"):
        """Generate visualizations of performance metrics.
        
        Args:
            metrics: The metrics dictionary
            output_dir: Directory to save visualizations
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Plot 1: Success rates by category
        self._plot_category_metrics(metrics, output_dir)
        
        # Plot 2: Average verification/repair times
        self._plot_timing_metrics(metrics, output_dir)
        
        # Save metrics as JSON
        with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
            # Convert sets to lists for JSON serialization
            serializable_metrics = self._make_json_serializable(metrics)
            json.dump(serializable_metrics, f, indent=2)
    
    def _make_json_serializable(self, obj):
        """Convert objects to JSON serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(i) for i in obj]
        elif isinstance(obj, (set, tuple)):
            return list(obj)
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    def _plot_category_metrics(self, metrics: Dict[str, Any], output_dir: str):
        """Plot metrics by category."""
        categories = list(metrics["by_category"].keys())
        success_rates = [metrics["by_category"][c]["success_rate"] for c in categories]
        detection_rates = [metrics["by_category"][c]["detection_rate"] for c in categories]
        repair_rates = [metrics["by_category"][c]["repair_success_rate"] for c in categories]
        
        # Format category names for display
        display_categories = [c.replace('_', '\n') for c in categories]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(categories))
        width = 0.25
        
        ax.bar(x - width, success_rates, width, label='Success Rate')
        ax.bar(x, detection_rates, width, label='Detection Rate')
        ax.bar(x + width, repair_rates, width, label='Repair Rate')
        
        ax.set_ylabel('Rate')
        ax.set_title('Performance by Logical Category')
        ax.set_xticks(x)
        ax.set_xticklabels(display_categories)
        ax.legend()
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "performance_by_category.png"))
        plt.close()
    
    def _plot_timing_metrics(self, metrics: Dict[str, Any], output_dir: str):
        """Plot timing metrics."""
        categories = list(metrics["by_category"].keys())
        verification_times = [statistics.mean(metrics["by_category"][c]["verification_times"]) 
                             if metrics["by_category"][c]["verification_times"] else 0 
                             for c in categories]
        
        repair_times = [statistics.mean(metrics["by_category"][c]["repair_times"]) 
                       if metrics["by_category"][c]["repair_times"] else 0 
                       for c in categories]
        
        # Format category names for display
        display_categories = [c.replace('_', '\n') for c in categories]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(categories))
        width = 0.35
        
        ax.bar(x - width/2, verification_times, width, label='Verification Time (s)')
        ax.bar(x + width/2, repair_times, width, label='Repair Time (s)')
        
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Average Processing Time by Category')
        ax.set_xticks(x)
        ax.set_xticklabels(display_categories)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "timing_by_category.png"))
        plt.close()

def main():
    """Run performance metrics collection and visualization."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Measure performance metrics for the ConsistencyVerifier')
    parser.add_argument('--test-cases', '-t', help='Path to test cases JSON file')
    parser.add_argument('--iterations', '-i', type=int, default=3, help='Number of iterations for each test case')
    parser.add_argument('--output-dir', '-o', default='results', help='Directory to save results')
    
    args = parser.parse_args()
    
    logger.info(f"Starting performance metrics collection with {args.iterations} iterations")
    
    metrics_collector = PerformanceMetrics(args.test_cases)
    metrics = metrics_collector.measure_performance(args.iterations)
    
    logger.info("Performance metrics collected")
    logger.info(f"Overall success rate: {metrics['success_rate']:.2f}")
    logger.info(f"Detection rate: {metrics['inconsistency_detection_rate']:.2f}")
    logger.info(f"Repair success rate: {metrics['repair_success_rate']:.2f}")
    logger.info(f"Average verification time: {metrics.get('avg_verification_time', 0):.2f}s")
    logger.info(f"Average repair time: {metrics.get('avg_repair_time', 0):.2f}s")
    
    metrics_collector.visualize_results(metrics, args.output_dir)
    logger.info(f"Results saved to {args.output_dir} directory")

if __name__ == "__main__":
    main() 