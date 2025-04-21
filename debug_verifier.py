#!/usr/bin/env python3
import os
import sys
from llm_consistency_verifier.core.enhanced_consistency_verifier import EnhancedConsistencyVerifier
from llm_consistency_verifier.utils.ontology_manager import OntologyManager

def main():
    # Check for OpenAI API key
    if "OPENAI_API_KEY" not in os.environ:
        print("ERROR: OPENAI_API_KEY environment variable not found!")
        print("Please set your OpenAI API key before running this script:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    # Initialize the verifier
    print("Initializing EnhancedConsistencyVerifier...")
    verifier = EnhancedConsistencyVerifier()
    
    # Print all attributes
    print("\nEnhancedConsistencyVerifier attributes before verification:")
    for attr in sorted(dir(verifier)):
        if not attr.startswith("__"):
            print(f"  - {attr}")
    
    # Run a verification to ensure all components are initialized
    test_text = "All animals are living things. All cats are animals. Fluffy is a cat."
    print(f"\nRunning verification on test text: {test_text}")
    result = verifier.verify(test_text)
    print(f"Verification result: {'CONSISTENT' if result.is_consistent else 'INCONSISTENT'}")
    
    # Print all attributes again after verification
    print("\nEnhancedConsistencyVerifier attributes after verification:")
    for attr in sorted(dir(verifier)):
        if not attr.startswith("__"):
            print(f"  - {attr}")
    
    # Try to access the ontology manager
    print("\nAttempting to access ontology_manager:")
    if hasattr(verifier, 'ontology_manager'):
        print("  ✓ verifier.ontology_manager exists!")
        ont = verifier.ontology_manager
        print(f"  Entities: {len(ont.all_entities)}")
    else:
        print("  ✗ verifier.ontology_manager does not exist")
    
    # Try to access enhanced_engine
    print("\nAttempting to access enhanced_engine:")
    if hasattr(verifier, 'enhanced_engine'):
        print("  ✓ verifier.enhanced_engine exists!")
        
        if hasattr(verifier.enhanced_engine, 'ontology_manager'):
            print("  ✓ verifier.enhanced_engine.ontology_manager exists!")
            ont = verifier.enhanced_engine.ontology_manager
            print(f"  Entities: {len(ont.all_entities)}")
        else:
            print("  ✗ verifier.enhanced_engine.ontology_manager does not exist")
    else:
        print("  ✗ verifier.enhanced_engine does not exist")
    
    # Try other potential paths
    print("\nAttempting to access verification_engine:")
    if hasattr(verifier, 'verification_engine'):
        print("  ✓ verifier.verification_engine exists!")
        
        if hasattr(verifier.verification_engine, 'ontology_manager'):
            print("  ✓ verifier.verification_engine.ontology_manager exists!")
            ont = verifier.verification_engine.ontology_manager
            print(f"  Entities: {len(ont.all_entities)}")
        else:
            print("  ✗ verifier.verification_engine.ontology_manager does not exist")
    else:
        print("  ✗ verifier.verification_engine does not exist")
    
    print("\nDebug complete.")

if __name__ == "__main__":
    main() 