#!/usr/bin/env python3
"""
Script to run evaluations with optimized API rate limiting.
This helps prevent hitting OpenAI rate limits during evaluation.
"""
import os
import sys
import time
import logging
import argparse
from pathlib import Path

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation.run_all_evaluations import run_all_evaluations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_optimized_evaluation(
    output_dir="results",
    api_rate=10,
    max_repair=1,
    iterations=1,
    use_disk_cache=True,
    test_cases=None,
    benchmark_file=None,
    clear_cache=False
):
    """
    Run evaluations with optimized settings to avoid rate limits.
    
    Args:
        output_dir: Directory to save results
        api_rate: Maximum API calls per minute
        max_repair: Maximum repair attempts per inconsistency
        iterations: Number of iterations to run for each test
        use_disk_cache: Use disk cache for API responses
        test_cases: Path to test cases JSON file
        benchmark_file: Path to benchmark JSON file
        clear_cache: Whether to clear the cache before running
    """
    # Set environment variables for optimal rate limiting
    os.environ["API_CALLS_PER_MINUTE"] = str(api_rate)
    os.environ["MAX_REPAIR_ATTEMPTS"] = str(max_repair)
    os.environ["USE_DISK_CACHE"] = str(use_disk_cache).lower()
    os.environ["CACHE_RESULTS"] = "True"
    
    # Clear cache if requested
    if clear_cache and use_disk_cache:
        cache_dir = Path(os.getenv("DISK_CACHE_DIR", ".cache"))
        if cache_dir.exists():
            logger.info(f"Clearing cache directory: {cache_dir}")
            for cache_file in cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete cache file {cache_file}: {str(e)}")
    
    # Set up output directory
    results_dir = Path(output_dir)
    if not results_dir.exists():
        results_dir.mkdir(parents=True)
    
    logger.info("Starting optimized evaluation run with rate limiting")
    logger.info(f"API Rate: {api_rate} calls per minute")
    logger.info(f"Max Repair Attempts: {max_repair}")
    logger.info(f"Iterations: {iterations}")
    logger.info(f"Disk Cache: {'Enabled' if use_disk_cache else 'Disabled'}")
    
    # Run the evaluation
    start_time = time.time()
    try:
        results = run_all_evaluations(
            output_dir=output_dir,
            iterations=iterations,
            test_cases=test_cases,
            benchmark_file=benchmark_file,
            api_calls_per_minute=api_rate,
            use_disk_cache=use_disk_cache,
            max_repair_attempts=max_repair
        )
        
        logger.info(f"Evaluation completed successfully in {time.time() - start_time:.2f}s")
        return results
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise

def main():
    """Handle command line arguments and run the evaluation."""
    parser = argparse.ArgumentParser(
        description='Run evaluations with optimized API rate limiting',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        default='results_optimized',
        help='Directory to save results'
    )
    
    parser.add_argument(
        '--api-rate', '-r',
        type=int,
        default=10,
        help='Maximum API calls per minute'
    )
    
    parser.add_argument(
        '--max-repair', '-m',
        type=int,
        default=1,
        help='Maximum repair attempts per inconsistency'
    )
    
    parser.add_argument(
        '--iterations', '-i',
        type=int,
        default=1,
        help='Number of iterations to run for each test'
    )
    
    parser.add_argument(
        '--no-cache', '-n',
        action='store_true',
        help='Disable disk caching of API responses'
    )
    
    parser.add_argument(
        '--clear-cache', '-c',
        action='store_true',
        help='Clear the disk cache before running'
    )
    
    parser.add_argument(
        '--test-cases', '-t',
        help='Path to test cases JSON file'
    )
    
    parser.add_argument(
        '--benchmark-file', '-b',
        help='Path to benchmark JSON file'
    )
    
    args = parser.parse_args()
    
    run_optimized_evaluation(
        output_dir=args.output_dir,
        api_rate=args.api_rate,
        max_repair=args.max_repair,
        iterations=args.iterations,
        use_disk_cache=not args.no_cache,
        test_cases=args.test_cases,
        benchmark_file=args.benchmark_file,
        clear_cache=args.clear_cache
    )

if __name__ == "__main__":
    main() 