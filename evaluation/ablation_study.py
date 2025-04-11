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

class AblationStudy:
    """Conducts ablation studies to analyze component effects."""
    
    def __init__(self, test_cases_path: str = None):
        """Initialize with test cases.
        
        Args:
            test_cases_path: Path to test cases JSON file (optional)
        """
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
        else:
            # Default test cases
            self.test_cases = [
                {
                    "text": "All birds can fly. Penguins are birds. Penguins cannot fly.",
                    "is_consistent": False
                },
                {
                    "text": "All mammals have fur. Cats are mammals. Cats have fur.",
                    "is_consistent": True
                },
                {
                    "text": "If a student studies, they will pass the exam. If they pass the exam, they will graduate. Alice studied but did not graduate.",
                    "is_consistent": False
                },
                {
                    "text": "Every integer is either even or odd. There exists an integer that is both even and odd.",
                    "is_consistent": False
                },
                {
                    "text": "If it rains, the ground gets wet. If the ground is wet, then it has rained.",
                    "is_consistent": False
                }
            ]
    
    def run_ablation_study(self) -> Dict[str, Any]:
        """Run ablation study with different configurations.
        
        Returns:
            Dictionary of results by configuration
        """
        # Base configuration
        base_config = {
            "model": Config.LLM_MODEL,
            "solver": Config.SOLVER_TYPE,
            "repair_attempts": Config.MAX_REPAIR_ATTEMPTS
        }
        
        # Configurations to test
        configurations = [
            {
                "name": "Base", 
                "model": base_config["model"], 
                "solver": base_config["solver"], 
                "repair_attempts": base_config["repair_attempts"]
            },
            {
                "name": "Alternative Model", 
                "model": "gpt-3.5-turbo", 
                "solver": base_config["solver"], 
                "repair_attempts": base_config["repair_attempts"]
            },
            {
                "name": "Alternative Solver", 
                "model": base_config["model"], 
                "solver": "sympy", 
                "repair_attempts": base_config["repair_attempts"]
            },
            {
                "name": "More Repair Attempts", 
                "model": base_config["model"], 
                "solver": base_config["solver"], 
                "repair_attempts": 5
            }
        ]
        
        results = {}
        original_repair_attempts = Config.MAX_REPAIR_ATTEMPTS
        
        for config in configurations:
            logger.info(f"Testing configuration: {config['name']}")
            
            # Create verifier with this configuration
            verifier = ConsistencyVerifier(
                llm_model=config["model"],
                solver_type=config["solver"]
            )
            
            # Update repair attempts config
            Config.MAX_REPAIR_ATTEMPTS = config["repair_attempts"]
            
            config_results = {
                "success_rate": 0,
                "detection_rate": 0,
                "repair_success_rate": 0,
                "avg_verification_time": 0,
                "avg_repair_time": 0,
                "inconsistencies_found": 0,
                "total_cases": len(self.test_cases)
            }
            
            detection_count = 0
            repair_count = 0
            
            for case in self.test_cases:
                ground_truth = case.get("is_consistent", True)
                
                # Verify
                start = time.time()
                result = verifier.verify(case["text"])
                verification_time = time.time() - start
                
                # Update metrics
                config_results["avg_verification_time"] += verification_time
                if result.is_consistent == ground_truth:
                    config_results["success_rate"] += 1
                
                # Inconsistency detection
                if not result.is_consistent:
                    detection_count += 1
                    config_results["inconsistencies_found"] += len(result.inconsistencies)
                    
                    # Try repair if inconsistent
                    start = time.time()
                    repaired = verifier.repair(case["text"])
                    repair_time = time.time() - start
                    
                    # Check if repair worked
                    repair_result = verifier.verify(repaired)
                    if repair_result.is_consistent:
                        repair_count += 1
                    
                    config_results["avg_repair_time"] += repair_time
            
            # Calculate averages
            if config_results["total_cases"] > 0:
                config_results["success_rate"] /= config_results["total_cases"]
                config_results["avg_verification_time"] /= config_results["total_cases"]
                
                if detection_count > 0:
                    config_results["detection_rate"] = detection_count / config_results["total_cases"]
                    config_results["avg_repair_time"] /= detection_count
                    config_results["repair_success_rate"] = repair_count / detection_count if detection_count > 0 else 0
                    config_results["inconsistencies_found"] /= detection_count
            
            results[config["name"]] = config_results
            logger.info(f"Completed testing configuration: {config['name']}")
        
        # Restore original repair attempts config
        Config.MAX_REPAIR_ATTEMPTS = original_repair_attempts
        
        return results
    
    def visualize_results(self, results: Dict[str, Any], output_dir: str = "results"):
        """Generate visualizations of ablation study results.
        
        Args:
            results: The results dictionary
            output_dir: Directory to save visualizations
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Plot 1: Success & repair rates by configuration
        self._plot_success_rates(results, output_dir)
        
        # Plot 2: Average times by configuration
        self._plot_times(results, output_dir)
        
        # Save results as JSON
        with open(os.path.join(output_dir, "ablation_results.json"), 'w') as f:
            json.dump(results, f, indent=2)
    
    def _plot_success_rates(self, results: Dict[str, Any], output_dir: str):
        """Plot success rates by configuration."""
        configs = list(results.keys())
        success_rates = [results[c]["success_rate"] for c in configs]
        detection_rates = [results[c]["detection_rate"] for c in configs]
        repair_rates = [results[c]["repair_success_rate"] for c in configs]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(configs))
        width = 0.25
        
        ax.bar(x - width, success_rates, width, label='Success Rate')
        ax.bar(x, detection_rates, width, label='Detection Rate')
        ax.bar(x + width, repair_rates, width, label='Repair Success Rate')
        
        ax.set_ylabel('Rate')
        ax.set_title('Performance by Configuration')
        ax.set_xticks(x)
        ax.set_xticklabels(configs)
        ax.legend()
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "ablation_success_rates.png"))
        plt.close()
    
    def _plot_times(self, results: Dict[str, Any], output_dir: str):
        """Plot timing metrics by configuration."""
        configs = list(results.keys())
        verification_times = [results[c]["avg_verification_time"] for c in configs]
        repair_times = [results[c]["avg_repair_time"] for c in configs]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(configs))
        width = 0.35
        
        ax.bar(x - width/2, verification_times, width, label='Avg. Verification Time (s)')
        ax.bar(x + width/2, repair_times, width, label='Avg. Repair Time (s)')
        
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Average Processing Time by Configuration')
        ax.set_xticks(x)
        ax.set_xticklabels(configs)
        ax.legend()
        
        for i, value in enumerate(verification_times):
            ax.text(i - width/2, value + 0.1, f'{value:.2f}s', ha='center')
        
        for i, value in enumerate(repair_times):
            ax.text(i + width/2, value + 0.1, f'{value:.2f}s', ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "ablation_timing.png"))
        plt.close()

def main():
    """Run ablation study and visualization."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ablation study for the ConsistencyVerifier')
    parser.add_argument('--test-cases', '-t', help='Path to test cases JSON file')
    parser.add_argument('--output-dir', '-o', default='results', help='Directory to save results')
    
    args = parser.parse_args()
    
    logger.info("Starting ablation study")
    
    study = AblationStudy(args.test_cases)
    results = study.run_ablation_study()
    
    logger.info("Ablation study completed")
    for config, res in results.items():
        logger.info(f"\n{config}:")
        for metric, value in res.items():
            if isinstance(value, float):
                logger.info(f"  {metric}: {value:.4f}")
            else:
                logger.info(f"  {metric}: {value}")
    
    study.visualize_results(results, args.output_dir)
    logger.info(f"Results saved to {args.output_dir} directory")

if __name__ == "__main__":
    main() 