# Enhanced LLM Consistency Verifier

This document provides information about the Enhanced LLM Consistency Verifier, which builds upon the original implementation to provide more robust logical consistency verification capabilities.

## Overview

The Enhanced LLM Consistency Verifier extends the core functionality of the original verifier with:

1. **Advanced Rule Extraction**: Uses LLM-based parsing to extract complex logical structures beyond regex patterns
2. **Hierarchical Ontology Management**: Manages relationships between entities and properties with proper inheritance
3. **Enhanced Z3 Integration**: Provides better handling of complex logical statements and improved constraint generation
4. **Context-Aware Verification**: Maintains context across verification sessions for more accurate reasoning

## Installation

The Enhanced Consistency Verifier is included in the main package. Install using:

```bash
pip install -e .
```

## Components

### AdvancedRuleExtractor

The `AdvancedRuleExtractor` improves upon the original pattern-based extraction by:

- Using LLM-based parsing for complex logical structures
- Supporting a wider range of logical forms including modal, temporal, and quantified expressions
- Providing structured output for more accurate logical representation

### OntologyManager

The `OntologyManager` handles hierarchical relationships between entities:

- Manages `is-a`, `part-of`, and `instance-of` relationships
- Handles property inheritance with exceptions
- Provides consistency checking for the ontology itself

### EnhancedVerificationEngine

The `EnhancedVerificationEngine` provides more sophisticated verification:

- Better integration with Z3 solver for complex constraint handling
- Improved detection of logical contradictions
- Support for more varied logical relationship types

### EnhancedConsistencyVerifier

The main interface for the enhanced verification system:

- Maintains backward compatibility with the original verifier
- Integrates all enhanced components
- Provides detailed logical analysis and explanations

## Usage Examples

### Basic Usage

```python
from llm_consistency_verifier.core.enhanced_consistency_verifier import EnhancedConsistencyVerifier

# Initialize the enhanced verifier
verifier = EnhancedConsistencyVerifier()

# Verify text for logical consistency
text = "All birds can fly. Penguins are birds. Penguins cannot fly."
result = verifier.verify(text)

# Check the result
if result.is_consistent:
    print("The text is logically consistent.")
else:
    print("The text contains logical inconsistencies:")
    for inconsistency in result.inconsistencies:
        print(f"- {inconsistency}")

# Attempt to repair inconsistent text
if not result.is_consistent:
    repaired_text = verifier.repair(text)
    print(f"Repaired text: {repaired_text}")
```

### Advanced Usage with Ontology Management

```python
from llm_consistency_verifier.core.enhanced_consistency_verifier import EnhancedConsistencyVerifier
from llm_consistency_verifier.utils.ontology_manager import OntologyManager

# Create and initialize ontology
ontology = OntologyManager()
ontology.add_entity("animal")
ontology.add_entity("bird", parent="animal")
ontology.add_entity("penguin", parent="bird")
ontology.add_property("can_fly")
ontology.add_property("has_feathers")

# Add property relationships
ontology.add_property_to_entity("bird", "has_feathers")
ontology.add_property_to_entity("bird", "can_fly")
ontology.add_property_exception("penguin", "can_fly", False)

# Initialize verifier with custom ontology
verifier = EnhancedConsistencyVerifier(ontology=ontology)

# Verify text with ontology-aware processing
text = "Penguins cannot fly. Tweety is a bird. Tweety can fly."
result = verifier.verify(text)
```

### Using LLM-Based Rule Extraction

```python
from llm_consistency_verifier.core.enhanced_consistency_verifier import EnhancedConsistencyVerifier

# Initialize with LLM-based rule extraction
verifier = EnhancedConsistencyVerifier(use_llm_extraction=True)

# This handles complex logical forms better
complex_text = """
If it rains, then the ground gets wet.
If the ground is wet, then the grass is slippery.
It rained yesterday, but the grass was not slippery.
"""

result = verifier.verify(complex_text)
```

## Command Line Interface

The enhanced verifier can also be used from the command line:

```bash
python -m llm_consistency_verifier.cli_enhanced --input "All birds can fly. Penguins are birds. Penguins cannot fly." --use-enhanced
```

Options:
- `--use-enhanced`: Use the enhanced verifier (default: False)
- `--use-llm-extraction`: Use LLM-based rule extraction (default: False)
- `--repair`: Attempt to repair inconsistencies (default: False)
- `--explain`: Provide detailed explanations (default: False)

## Comparison with Standard Verifier

The enhanced verifier provides several advantages over the standard implementation:

1. **Better handling of complex logical statements**: Captures relationships that the standard pattern-based approach might miss
2. **More accurate inconsistency detection**: Particularly for nested logical structures and hierarchical relationships
3. **Improved repair capabilities**: More nuanced repairs that respect logical coherence
4. **Detailed explanations**: Provides more insightful analysis of logical inconsistencies

## Limitations

While the enhanced verifier improves upon the original implementation, it still has some limitations:

1. **Computational complexity**: More sophisticated verification can require more processing time
2. **LLM dependency**: Advanced rule extraction relies on LLM services
3. **Domain specificity**: May require domain-specific knowledge for specialized texts

## Future Work

Planned enhancements include:

1. **Domain-specific ontology support**: Loading pre-defined ontologies for specific domains
2. **Multi-document consistency verification**: Checking consistency across multiple related texts
3. **Interactive inconsistency resolution**: Guided interface for resolving logical contradictions
4. **Natural language explanation improvement**: More human-readable explanations of logical issues

## Contributing

Contributions to the Enhanced LLM Consistency Verifier are welcome. Please see the main repository README for contribution guidelines. 