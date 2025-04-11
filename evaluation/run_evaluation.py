#!/usr/bin/env python3
import os
import sys
import time
import logging
from pathlib import Path

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation.performance_metrics import PerformanceMetrics
from evaluation.benchmark_suite import BenchmarkSuite
from evaluation.ablation_study import AblationStudy
from evaluation.baseline_comparison import BaselineComparison

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Run comprehensive evaluation of the LLM consistency verifier."""
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Load test cases
    test_cases_path = Path("evaluation/test_cases.json")
    if not test_cases_path.exists():
        logger.error(f"Test cases file not found: {test_cases_path}")
        sys.exit(1)
    
    logger.info("Starting comprehensive evaluation")
    start_time = time.time()
    
    # 1. Run performance metrics
    logger.info("Running performance metrics...")
    metrics = PerformanceMetrics(str(test_cases_path))
    performance_results = metrics.measure_performance(iterations=5)
    metrics.visualize_results(performance_results, str(results_dir))
    
    # 2. Run benchmark suite
    logger.info("Running benchmark suite...")
    benchmark = BenchmarkSuite(str(test_cases_path))
    benchmark_results = benchmark.run_benchmark()
    benchmark.visualize_results(benchmark_results, str(results_dir))
    
    # 3. Run ablation study
    logger.info("Running ablation study...")
    ablation = AblationStudy(str(test_cases_path))
    ablation_results = ablation.run_ablation_study()
    ablation.visualize_results(ablation_results, str(results_dir))
    
    # 4. Run baseline comparison
    logger.info("Running baseline comparison...")
    comparison = BaselineComparison(str(test_cases_path))
    comparison_results = comparison.run_comparison()
    comparison.visualize_results(comparison_results, str(results_dir))
    
    # Print summary
    total_time = time.time() - start_time
    logger.info(f"\nEvaluation completed in {total_time:.2f} seconds")
    logger.info(f"Results saved to {results_dir}")
    
    # Print key metrics
    logger.info("\nKey Performance Metrics:")
    logger.info(f"Success Rate: {performance_results['success_rate']:.2%}")
    logger.info(f"Detection Rate: {performance_results['inconsistency_detection_rate']:.2%}")
    logger.info(f"Repair Success Rate: {performance_results['repair_success_rate']:.2%}")
    logger.info(f"Average Verification Time: {performance_results.get('avg_verification_time', 0):.2f}s")
    logger.info(f"Average Repair Time: {performance_results.get('avg_repair_time', 0):.2f}s")

if __name__ == "__main__":
    main() 