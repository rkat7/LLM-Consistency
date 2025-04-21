# LLM Consistency Verifier Workflow

This document outlines the complete workflow of the LLM Consistency Verifier system, showing how text is processed from input to verification results.

## Overview Diagram

```
┌─────────────────┐     ┌─────────────────┐     ┌────────────────────┐     ┌───────────────┐
│                 │     │                 │     │                    │     │               │
│   Text Input    │────▶│ Rule Extraction │────▶│ Formal Verification│────▶│    Results    │
│                 │     │                 │     │                    │     │               │
└─────────────────┘     └─────────────────┘     └────────────────────┘     └───────────────┘
                                                                                   │
                                                                                   │
                                                                                   ▼
                                                                          ┌───────────────┐
                                                                          │               │
                                                                          │  (Optional)   │
                                                                          │    Repair     │
                                                                          │               │
                                                                          └───────────────┘
```

## Detailed Process Flow

### 1. Text Input

The system receives natural language text from the user that needs to be checked for logical consistency.

- Input types: documents, paragraphs, statements, or other forms of text
- Cache checking: The system first checks if this text has been processed before to avoid redundant work

### 2. Rule Extraction

The system extracts formal logical rules from the natural language text.

**Two extraction methods:**

1. **Pattern-based extraction** (Primary method)
   - Uses regular expressions and rule-based patterns
   - Extracts different types of logical rules:
     - Universal statements (All X are Y)
     - Existential statements (Some X are Y)
     - Implications (If X then Y)
     - Negations (X is not Y)
     - Assertions (X is Y)

2. **LLM-assisted extraction** (Fallback method)
   - Uses large language models to extract rules when pattern matching is insufficient
   - More powerful but slower and more resource-intensive
   - Automatically used when pattern-based extraction finds few rules

### 3. Formal Verification

The extracted rules are formalized and verified for logical consistency.

**Formalization process:**
- Normalizes propositions (standardizing language and format)
- Maps text to logical variables
- Creates formal logical constraints

**Verification methods:**
- **Z3 Solver** (Primary method): Uses Microsoft's Z3 SMT solver to check satisfiability
- **SymPy** (Alternative method): Algebraic approach to consistency checking

**Verification outcome:**
- **Consistent**: No logical contradictions found
- **Inconsistent**: Logical contradictions detected, with specific inconsistencies identified

### 4. Results

The system returns the verification results to the user.

**Result components:**
- Consistency status (consistent/inconsistent)
- List of inconsistencies (if any)
- Verification time and performance metrics

### 5. Repair (Optional)

If inconsistencies are found, the system can attempt to repair the text.

**Repair process:**
1. Send inconsistencies and original text to LLM
2. Generate repaired text that resolves inconsistencies
3. Verify the repaired text
4. Repeat if necessary (up to configured max attempts)
5. Return the best version (fewest inconsistencies)

## Logging Flow

Throughout the process, detailed flow logging occurs with the following format:

- `[FLOW:STEP 1]` - Text input reception
- `[FLOW:STEP 2]` - Rule extraction
- `[FLOW:STEP 3]` - Formal verification
- `[FLOW:STEP 4]` - Results generation

For the repair process:
- `[FLOW:REPAIR:STEP 1]` - Initial verification
- `[FLOW:REPAIR:STEP 2]` - Repair preparation
- `[FLOW:REPAIR:STEP 3]` - Iterative repair
- `[FLOW:REPAIR:STEP 4]` - Finalization

## Example Workflow

```
Text Input: "All birds can fly. Penguins are birds. Penguins cannot fly."
    ↓
Rule Extraction:
    - "All birds can fly" (Universal)
    - "Penguins are birds" (Assertion)
    - "Penguins cannot fly" (Negation)
    ↓
Formalization:
    - birds_can_fly
    - penguins_are_birds
    - NOT(penguins_can_fly)
    ↓
Verification (Z3 Solver):
    - Contradiction: Universal rule (all birds can fly) + assertion (penguins are birds) 
      implies penguins can fly, but negation (penguins cannot fly) contradicts this
    ↓
Results:
    - Inconsistent
    - Inconsistency: Contradiction between "All birds can fly", "Penguins are birds", and "Penguins cannot fly"
    ↓
Repair:
    - Add exceptions: "Almost all birds can fly. Penguins are birds. Penguins cannot fly."
    ↓
Final Output:
    - Consistent repaired text
``` 