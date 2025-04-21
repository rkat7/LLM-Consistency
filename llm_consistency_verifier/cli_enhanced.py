#!/usr/bin/env python3
"""
Enhanced CLI for LLM Consistency Verifier.
This module provides an enhanced command-line interface for verifying the logical consistency
of text using the EnhancedConsistencyVerifier.
"""

import argparse
import sys
import logging
import json
import os
from pathlib import Path

from llm_consistency_verifier.core.consistency_verifier import ConsistencyVerifier
from llm_consistency_verifier.core.enhanced_consistency_verifier import EnhancedConsistencyVerifier
from llm_consistency_verifier.utils.ontology_manager import OntologyManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def setup_argparse():
    """Set up argument parsing for the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Verify logical consistency of text using enhanced LLM Consistency Verifier."
    )
    
    # Input sources
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-i", "--input", 
        help="Input text to verify for logical consistency"
    )
    input_group.add_argument(
        "-f", "--file", 
        help="Path to file containing text to verify"
    )
    
    # Output options
    parser.add_argument(
        "-o", "--output", 
        help="Output file to write verification results (defaults to stdout)"
    )
    parser.add_argument(
        "--json", 
        action="store_true", 
        help="Output results in JSON format"
    )
    
    # Verification options
    parser.add_argument(
        "--use-enhanced", 
        action="store_true", 
        help="Use enhanced verification engine (default: False)"
    )
    parser.add_argument(
        "--use-llm-extraction", 
        action="store_true", 
        help="Use LLM-based rule extraction (default: False)"
    )
    parser.add_argument(
        "--ontology-file", 
        help="Path to JSON file containing predefined ontology"
    )
    parser.add_argument(
        "--repair", 
        action="store_true", 
        help="Attempt to repair logical inconsistencies"
    )
    parser.add_argument(
        "--repair-attempts", 
        type=int, 
        default=3, 
        help="Maximum number of repair attempts (default: 3)"
    )
    parser.add_argument(
        "--explain", 
        action="store_true", 
        help="Provide detailed explanations of inconsistencies"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose logging"
    )
    
    return parser

def load_ontology(ontology_file):
    """Load a predefined ontology from a JSON file."""
    logger.info(f"Loading ontology from {ontology_file}")
    try:
        with open(ontology_file, 'r') as f:
            ontology_data = json.load(f)
        
        ontology = OntologyManager()
        
        # Load entities
        for entity_data in ontology_data.get('entities', []):
            entity_name = entity_data['name']
            parent = entity_data.get('parent', None)
            relationship = entity_data.get('relationship', 'is-a')
            ontology.add_entity(entity_name, parent, relationship)
        
        # Load properties
        for property_data in ontology_data.get('properties', []):
            property_name = property_data['name']
            ontology.add_property(property_name)
        
        # Load property assignments
        for assignment in ontology_data.get('property_assignments', []):
            entity = assignment['entity']
            property_name = assignment['property']
            value = assignment.get('value', True)
            is_exception = assignment.get('is_exception', False)
            
            if is_exception:
                ontology.add_property_exception(entity, property_name, value)
            else:
                ontology.add_property_to_entity(entity, property_name, value)
        
        logger.info(f"Successfully loaded ontology with {len(ontology_data.get('entities', []))} entities")
        return ontology
    
    except Exception as e:
        logger.error(f"Failed to load ontology: {e}")
        return None

def format_output(result, text, repaired_text=None, json_output=False):
    """Format the verification result for output."""
    if json_output:
        output_data = {
            "is_consistent": result.is_consistent,
            "input_text": text,
            "inconsistencies": [str(inconsistency) for inconsistency in result.inconsistencies],
            "verification_time": result.verification_time,
        }
        
        if repaired_text:
            output_data["repaired_text"] = repaired_text
            
        return json.dumps(output_data, indent=2)
    else:
        # Plain text output
        output = []
        output.append("== LLM Consistency Verification Result ==")
        output.append(f"Consistency: {'✅ CONSISTENT' if result.is_consistent else '❌ INCONSISTENT'}")
        output.append(f"Verification time: {result.verification_time:.4f} seconds")
        
        if not result.is_consistent:
            output.append("\nLogical inconsistencies found:")
            for i, inconsistency in enumerate(result.inconsistencies, 1):
                output.append(f"  {i}. {inconsistency}")
        
        if repaired_text:
            output.append("\n== Repaired Text ==")
            output.append(repaired_text)
            
        return "\n".join(output)

def main():
    """Run the enhanced CLI for LLM Consistency Verifier."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Configure verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # Read input text
    if args.file:
        logger.info(f"Reading input from file: {args.file}")
        try:
            with open(args.file, 'r') as f:
                text = f.read()
        except Exception as e:
            logger.error(f"Failed to read input file: {e}")
            sys.exit(1)
    else:
        logger.info("Using input text from command line")
        text = args.input
    
    logger.info(f"Input text length: {len(text)} characters")
    
    # Load ontology if specified
    ontology = None
    if args.ontology_file:
        ontology = load_ontology(args.ontology_file)
        if not ontology:
            logger.warning("Proceeding without custom ontology")
    
    # Initialize verifier
    try:
        if args.use_enhanced:
            logger.info("Initializing EnhancedConsistencyVerifier")
            verifier = EnhancedConsistencyVerifier(
                use_llm_extraction=args.use_llm_extraction,
                ontology=ontology,
            )
        else:
            logger.info("Initializing standard ConsistencyVerifier")
            verifier = ConsistencyVerifier()
        
        logger.info("Verifying text for logical consistency")
        start_time = __import__('time').time()
        result = verifier.verify(text)
        result.verification_time = __import__('time').time() - start_time
        
        # Attempt repair if requested
        repaired_text = None
        if args.repair and not result.is_consistent:
            logger.info("Attempting to repair logical inconsistencies")
            repaired_text = verifier.repair(
                text, 
                max_attempts=args.repair_attempts
            )
            logger.info("Repair completed")
        
        # Format and output results
        output = format_output(result, text, repaired_text, args.json)
        
        if args.output:
            logger.info(f"Writing output to file: {args.output}")
            with open(args.output, 'w') as f:
                f.write(output)
        else:
            print(output)
        
        # Return exit code based on consistency
        sys.exit(0 if result.is_consistent else 1)
        
    except Exception as e:
        logger.error(f"Error during verification: {e}", exc_info=True)
        sys.exit(2)

if __name__ == "__main__":
    main() 