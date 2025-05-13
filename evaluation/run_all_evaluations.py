#!/usr/bin/env python3
import os
import sys
import time
import logging
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation.performance_metrics import PerformanceMetrics
from evaluation.ablation_study import AblationStudy
from evaluation.benchmark_suite import BenchmarkSuite
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

def run_all_evaluations(output_dir="results", iterations=3, test_cases=None, benchmark_file=None, 
                        api_calls_per_minute=None, use_disk_cache=None, max_repair_attempts=None):
    """Run all evaluation methods and generate a combined report.
    
    Args:
        output_dir: Directory to save results
        iterations: Number of iterations to run for each test
        test_cases: Path to test cases JSON file (optional)
        benchmark_file: Path to benchmark JSON file (optional)
        api_calls_per_minute: Override API rate limit setting (optional)
        use_disk_cache: Override disk cache setting (optional)
        max_repair_attempts: Override maximum repair attempts (optional)
    """
    start_time = time.time()
    logger.info("Starting comprehensive evaluation")
    
    # Override environment variables if specified
    if api_calls_per_minute is not None:
        os.environ["API_CALLS_PER_MINUTE"] = str(api_calls_per_minute)
        logger.info(f"Setting API call rate limit to {api_calls_per_minute} calls per minute")
    
    if use_disk_cache is not None:
        os.environ["USE_DISK_CACHE"] = str(use_disk_cache)
        logger.info(f"Setting disk cache to {use_disk_cache}")
    
    if max_repair_attempts is not None:
        os.environ["MAX_REPAIR_ATTEMPTS"] = str(max_repair_attempts)
        logger.info(f"Setting max repair attempts to {max_repair_attempts}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Run performance metrics
    logger.info("1. Running performance metrics evaluation")
    metrics_collector = PerformanceMetrics(test_cases)
    performance_metrics = metrics_collector.measure_performance(iterations)
    metrics_collector.visualize_results(performance_metrics, output_dir)
    
    # Run ablation study
    logger.info("2. Running ablation study")
    study = AblationStudy(test_cases)
    ablation_results = study.run_ablation_study()
    study.visualize_results(ablation_results, output_dir)
    
    # Run benchmark suite
    logger.info("3. Running benchmark suite")
    suite = BenchmarkSuite(benchmark_file)
    benchmark_results = suite.run_benchmark()
    suite.visualize_results(benchmark_results, output_dir)
    
    # Run baseline comparison
    logger.info("4. Running baseline comparison")
    comparison = BaselineComparison(test_cases)
    comparison_results = comparison.run_comparison()
    comparison.visualize_results(comparison_results, output_dir)
    
    # Generate combined summary
    logger.info("5. Generating combined summary")
    generate_summary_report(
        performance_metrics, 
        ablation_results, 
        benchmark_results, 
        comparison_results, 
        output_dir
    )
    
    total_time = time.time() - start_time
    logger.info(f"Comprehensive evaluation completed in {total_time:.2f}s")
    logger.info(f"All results saved to {output_dir} directory")
    
    # Return a summary of results
    return {
        "performance_metrics": performance_metrics,
        "ablation_results": ablation_results,
        "benchmark_results": benchmark_results,
        "comparison_results": comparison_results,
        "total_time": total_time
    }

def generate_summary_report(performance_metrics, ablation_results, benchmark_results, comparison_results, output_dir):
    """Generate a summary report of all evaluations.
    
    Args:
        performance_metrics: Results from performance metrics evaluation
        ablation_results: Results from ablation study
        benchmark_results: Results from benchmark suite
        comparison_results: Results from baseline comparison
        output_dir: Directory to save the report
    """
    # Create summary plot
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Overall performance metrics
    axs[0, 0].bar(['Success Rate', 'Detection Rate', 'Repair Rate'], 
              [performance_metrics.get('success_rate', 0.0), 
               performance_metrics.get('inconsistency_detection_rate', 0.0), 
               performance_metrics.get('repair_success_rate', 0.0)])
    axs[0, 0].set_title('Overall Performance Metrics')
    axs[0, 0].set_ylim(0, 1)
    
    # Plot 2: Comparison with baseline
    methods = ["Baseline", "Neural-Symbolic"]
    accuracy = [comparison_results.get("baseline", {}).get("accuracy", 0.0), 
                comparison_results.get("verifier", {}).get("accuracy", 0.0)]
    axs[0, 1].bar(methods, accuracy, color=['lightgray', 'green'])
    axs[0, 1].set_title('Accuracy Comparison')
    axs[0, 1].set_ylim(0, 1)
    
    # Plot 3: Benchmark results by category
    categories = [c for c in benchmark_results.keys() if c != "overall"]
    detection_rates = [benchmark_results.get(c, {}).get("detection_rate", 0.0) for c in categories]
    repair_rates = [benchmark_results.get(c, {}).get("repair_success_rate", 0.0) for c in categories]
    
    x = np.arange(len(categories))
    width = 0.35
    axs[1, 0].bar(x - width/2, detection_rates, width, label='Detection')
    axs[1, 0].bar(x + width/2, repair_rates, width, label='Repair')
    axs[1, 0].set_title('Benchmark Results by Category')
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels([c.replace('_', '\n') for c in categories])
    axs[1, 0].legend()
    axs[1, 0].set_ylim(0, 1)
    
    # Plot 4: Ablation study results
    configs = list(ablation_results.keys())
    success_rates = [ablation_results.get(c, {}).get("success_rate", 0.0) for c in configs]
    repair_rates = [ablation_results.get(c, {}).get("repair_success_rate", 0.0) for c in configs]
    
    x = np.arange(len(configs))
    width = 0.35
    axs[1, 1].bar(x - width/2, success_rates, width, label='Success')
    axs[1, 1].bar(x + width/2, repair_rates, width, label='Repair')
    axs[1, 1].set_title('Ablation Study Results')
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(configs)
    axs[1, 1].legend()
    axs[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_report.png"))
    plt.close()
    
    # Create a text summary report
    with open(os.path.join(output_dir, "summary_report.txt"), 'w') as f:
        f.write("LLM CONSISTENCY VERIFIER - EVALUATION SUMMARY\n")
        f.write("===========================================\n\n")
        
        f.write("1. OVERALL PERFORMANCE METRICS\n")
        f.write(f"Success Rate: {performance_metrics.get('success_rate', 0.0):.2f}\n")
        f.write(f"Detection Rate: {performance_metrics.get('inconsistency_detection_rate', 0.0):.2f}\n")
        f.write(f"Repair Success Rate: {performance_metrics.get('repair_success_rate', 0.0):.2f}\n")
        f.write(f"Average Verification Time: {performance_metrics.get('avg_verification_time', 0.0):.2f}s\n")
        f.write(f"Average Repair Time: {performance_metrics.get('avg_repair_time', 0.0):.2f}s\n\n")
        
        f.write("2. COMPARISON WITH BASELINE\n")
        f.write(f"Verifier Accuracy: {comparison_results.get('verifier', {}).get('accuracy', 0.0):.2f}\n")
        f.write(f"Baseline Accuracy: {comparison_results.get('baseline', {}).get('accuracy', 0.0):.2f}\n")
        f.write(f"Verifier F1 Score: {comparison_results.get('verifier', {}).get('f1', 0.0):.2f}\n")
        f.write(f"Baseline F1 Score: {comparison_results.get('baseline', {}).get('f1', 0.0):.2f}\n\n")
        
        f.write("3. BENCHMARK RESULTS\n")
        f.write(f"Overall Detection Rate: {benchmark_results.get('overall', {}).get('detection_rate', 0.0):.2f}\n")
        f.write(f"Overall Repair Success Rate: {benchmark_results.get('overall', {}).get('repair_success_rate', 0.0):.2f}\n")
        f.write("By Category:\n")
        for category in categories:
            f.write(f"  {category.replace('_', ' ').title()}:\n")
            f.write(f"    Detection Rate: {benchmark_results.get(category, {}).get('detection_rate', 0.0):.2f}\n")
            f.write(f"    Repair Success Rate: {benchmark_results.get(category, {}).get('repair_success_rate', 0.0):.2f}\n")
        f.write("\n")
        
        f.write("4. ABLATION STUDY RESULTS\n")
        for config in configs:
            f.write(f"{config}:\n")
            f.write(f"  Success Rate: {ablation_results.get(config, {}).get('success_rate', 0.0):.2f}\n")
            f.write(f"  Repair Success Rate: {ablation_results.get(config, {}).get('repair_success_rate', 0.0):.2f}\n")
            f.write(f"  Avg. Verification Time: {ablation_results.get(config, {}).get('avg_verification_time', 0.0):.2f}s\n")
        f.write("\n")
        
        f.write("5. LIMITATIONS AND FUTURE WORK\n")
        f.write("- Current implementation has limited support for complex logical structures\n")
        f.write("- Rule extraction from natural language needs improvement\n")
        f.write("- The repair mechanism could benefit from more formal guidance\n")
        f.write("- Future work should explore more advanced verification engines\n")

def main():
    """Run all evaluations with command line options."""
    parser = argparse.ArgumentParser(description='Run comprehensive evaluation of LLM Consistency Verifier')
    parser.add_argument('--test-cases', '-t', help='Path to test cases JSON file')
    parser.add_argument('--benchmark-file', '-b', help='Path to benchmark JSON file')
    parser.add_argument('--iterations', '-i', type=int, default=3, help='Number of iterations for each test case')
    parser.add_argument('--output-dir', '-o', default='results', help='Directory to save results')
    parser.add_argument('--api-rate', type=int, help='Maximum API calls per minute (default: 20)')
    parser.add_argument('--disk-cache', type=bool, default=True, help='Enable disk caching of API responses (default: True)')
    parser.add_argument('--max-repair', type=int, help='Maximum repair attempts per inconsistency (default: 3)')
    
    args = parser.parse_args()
    
    run_all_evaluations(
        output_dir=args.output_dir,
        iterations=args.iterations,
        test_cases=args.test_cases,
        benchmark_file=args.benchmark_file,
        api_calls_per_minute=args.api_rate,
        use_disk_cache=args.disk_cache,
        max_repair_attempts=args.max_repair
    )

if __name__ == "__main__":
    main() 