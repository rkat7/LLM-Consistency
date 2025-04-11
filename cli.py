#!/usr/bin/env python3
import argparse
import logging
import os
import sys
import time
from llm_consistency_verifier.core.consistency_verifier import ConsistencyVerifier
from llm_consistency_verifier.config.config import Config

def setup_logging():
    """Set up logging configuration."""
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    
    # Ensure logs directory exists
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # Configure logging
    log_file = os.path.join(logs_dir, "cli.log")
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def read_input_text(input_file=None):
    """Read input text from file or stdin."""
    if input_file:
        try:
            with open(input_file, 'r') as f:
                return f.read()
        except Exception as e:
            logging.error(f"Error reading file {input_file}: {str(e)}")
            sys.exit(1)
    else:
        # Check if input is being piped in
        if not sys.stdin.isatty():
            return sys.stdin.read()
        else:
            print("Enter text to verify (Ctrl+D to end input):")
            return sys.stdin.read()

def main():
    """Main CLI entrypoint."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='LLM Consistency Verifier CLI')
    parser.add_argument('--input', '-i', help='Input file to verify (default: stdin)')
    parser.add_argument('--repair', '-r', action='store_true', help='Attempt to repair inconsistencies')
    parser.add_argument('--explain', '-e', action='store_true', help='Explain inconsistencies in detail')
    parser.add_argument('--provider', '-p', help=f'LLM provider (default: {Config.LLM_PROVIDER})')
    parser.add_argument('--model', '-m', help=f'LLM model (default: {Config.LLM_MODEL})')
    parser.add_argument('--solver', '-s', help=f'Solver type (default: {Config.SOLVER_TYPE})')
    parser.add_argument('--output', '-o', help='Output file for results')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    logging.info("Starting LLM Consistency Verifier CLI")
    
    # Read input text
    input_text = read_input_text(args.input)
    if not input_text:
        logging.error("No input text provided")
        print("Error: No input text provided")
        sys.exit(1)
    
    # Initialize verifier with custom options if provided
    verifier_args = {}
    if args.provider:
        verifier_args['llm_provider'] = args.provider
    if args.model:
        verifier_args['llm_model'] = args.model
    if args.solver:
        verifier_args['solver_type'] = args.solver
    
    verifier = ConsistencyVerifier(**verifier_args)
    
    # Verify text
    start_time = time.time()
    verification_result = verifier.verify(input_text)
    
    # Prepare output
    if verification_result.is_consistent:
        result_text = "The text is logically consistent."
    else:
        result_text = f"Inconsistencies detected: {len(verification_result.inconsistencies)}\n"
        
        # Add detailed explanation if requested
        if args.explain:
            result_text += verifier.explain_inconsistencies(verification_result)
        else:
            # Simple list of inconsistencies
            for i, inconsistency in enumerate(verification_result.inconsistencies, 1):
                result_text += f"{i}. {inconsistency}\n"
        
        # Attempt repair if requested
        if args.repair:
            print("Attempting to repair inconsistencies...")
            repaired_text = verifier.repair(input_text)
            result_text += "\n--- Repaired Text ---\n"
            result_text += repaired_text
    
    # Add timing information
    result_text += f"\nProcessing time: {time.time() - start_time:.2f} seconds"
    
    # Output results
    if args.output:
        try:
            with open(args.output, 'w') as f:
                f.write(result_text)
            print(f"Results written to {args.output}")
        except Exception as e:
            logging.error(f"Error writing to output file: {str(e)}")
            print(f"Error writing to output file: {str(e)}")
            print(result_text)
    else:
        print("\n" + result_text)
    
    logging.info("CLI execution completed")

if __name__ == "__main__":
    main() 