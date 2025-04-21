#!/usr/bin/env python3
import os
import sys
import time
import json
import logging
from llm_consistency_verifier.core.enhanced_consistency_verifier import EnhancedConsistencyVerifier
from llm_consistency_verifier.core.consistency_verifier import ConsistencyVerifier
from llm_consistency_verifier.utils.ontology_manager import OntologyManager
from llm_consistency_verifier.utils.advanced_rule_extractor import AdvancedRuleExtractor

# ANSI color codes for better output formatting
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    PURPLE = '\033[95m'  # Added for implication rules

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}")

def print_success(text):
    print(f"{Colors.GREEN}{text}{Colors.ENDC}")

def print_warning(text):
    print(f"{Colors.YELLOW}{text}{Colors.ENDC}")

def print_error(text):
    print(f"{Colors.RED}{text}{Colors.ENDC}")

def print_info(text):
    print(f"{Colors.BLUE}{text}{Colors.ENDC}")

def print_code(text):
    print(f"{Colors.YELLOW}{text}{Colors.ENDC}")

# Custom logger to capture Enhanced Verifier internals
class InternalFlowCapture:
    def __init__(self, debug=False):
        self.messages = []
        self.debug = debug
        self.capture_handler = self._create_handler()
        self.original_level = None
    
    def _create_handler(self):
        handler = logging.StreamHandler(stream=self)
        handler.setLevel(logging.DEBUG if self.debug else logging.INFO)
        formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
        handler.setFormatter(formatter)
        return handler
    
    def write(self, message):
        if message.strip():
            self.messages.append(message.strip())
    
    def flush(self):
        pass
    
    def start_capture(self):
        self.messages = []
        # Get root logger
        root_logger = logging.getLogger()
        self.original_level = root_logger.level
        root_logger.setLevel(logging.DEBUG if self.debug else logging.INFO)
        root_logger.addHandler(self.capture_handler)
        
        # Also capture specific loggers we're interested in
        for logger_name in [
            'llm_consistency_verifier.core.enhanced_verification_engine',
            'llm_consistency_verifier.core.enhanced_consistency_verifier',
            'llm_consistency_verifier.utils.ontology_manager',
            'llm_consistency_verifier.utils.advanced_rule_extractor'
        ]:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.DEBUG if self.debug else logging.INFO)
    
    def stop_capture(self):
        # Get root logger
        root_logger = logging.getLogger()
        root_logger.removeHandler(self.capture_handler)
        root_logger.setLevel(self.original_level)
    
    def get_messages(self):
        return self.messages

def get_multiline_input():
    print_info("Enter your text for logical consistency verification.")
    print_info("Enter a blank line when you are done typing:")
    print("-------------------------------------------")

    lines = []
    while True:
        try:
            line = input()
            if line.strip() == '':
                break
            lines.append(line)
        except EOFError:
            break

    return '\n'.join(lines)

def display_internal_flow(flow_capture, detailed=False):
    print_header("INTERNAL VERIFICATION FLOW")
    
    # Filter and organize messages by phase
    ontology_msgs = []
    extraction_msgs = []
    z3_msgs = []
    general_msgs = []
    
    for msg in flow_capture.get_messages():
        if "ONTOLOGY" in msg:
            ontology_msgs.append(msg)
        elif "EXTRACT" in msg:
            extraction_msgs.append(msg)
        elif "Z3" in msg:
            z3_msgs.append(msg)
        else:
            general_msgs.append(msg)
    
    # Display extraction phase
    if extraction_msgs:
        print_info("⏳ RULE EXTRACTION PHASE")
        for msg in extraction_msgs:
            if detailed or "Successfully extracted" in msg:
                print(f"  {msg}")
    
    # Display ontology building phase  
    if ontology_msgs:
        print_info("🔄 ONTOLOGY BUILDING PHASE")
        for msg in ontology_msgs:
            if detailed or "Built ontology with" in msg:
                print(f"  {msg}")
    
    # Display Z3 solver phase
    if z3_msgs:
        print_info("🧮 Z3 SOLVER PHASE")
        for msg in z3_msgs:
            if detailed or any(x in msg for x in ["satisfiable", "contradiction", "pattern", "constraints"]):
                print(f"  {msg}")
    
    # Display other important messages
    if general_msgs and detailed:
        print_info("ℹ️ OTHER PROCESSES")
        for msg in general_msgs:
            if "inconsistency" in msg.lower() or "constraints" in msg.lower():
                print(f"  {msg}")

def display_ontology(verifier):
    print_header("ONTOLOGY VISUALIZATION")
    
    # Based on debug results, we now know the correct paths to the ontology manager
    if hasattr(verifier, 'enhanced_engine') and hasattr(verifier.enhanced_engine, 'ontology_manager'):
        # This is the correct path for EnhancedConsistencyVerifier
        ontology_manager = verifier.enhanced_engine.ontology_manager
    elif hasattr(verifier, '_verification_engine'):
        # For the standard ConsistencyVerifier
        ontology_manager = verifier._verification_engine.ontology_manager
    elif hasattr(verifier, 'ontology_manager'):
        # Direct access - may be empty
        ontology_manager = verifier.ontology_manager
    else:
        # Fallback if we can't find the ontology manager
        print_warning("  Could not access ontology from verifier. Creating a new visualization.")
        ontology_manager = OntologyManager()
    
    # Display entities and hierarchy
    print_info("📊 ENTITY HIERARCHY")
    
    # Get all entities
    entities = sorted(list(ontology_manager.all_entities))
    if not entities:
        print("  No entities found in ontology")
    else:
        # Build hierarchy tree
        hierarchy = {}
        for entity in entities:
            # Skip if this entity is a top-level entity with no parents
            if entity not in ontology_manager.is_a_relations:
                if entity not in hierarchy:
                    hierarchy[entity] = []
                continue
                
            # Add all parents
            for parent in ontology_manager.is_a_relations.get(entity, set()):
                if parent not in hierarchy:
                    hierarchy[parent] = []
                if entity not in hierarchy[parent]:
                    hierarchy[parent].append(entity)
        
        # Find root entities (those that are not children of any entity)
        children = set()
        for parent, child_list in hierarchy.items():
            children.update(child_list)
        
        roots = [e for e in entities if e not in children]
        
        # Display the hierarchy tree
        def print_tree(entity, indent=0, visited=None):
            if visited is None:
                visited = set()
                
            if entity in visited:
                return
                
            visited.add(entity)
            print(f"  {'  ' * indent}{'└─ ' if indent > 0 else ''}{entity}")
            
            for child in sorted(hierarchy.get(entity, [])):
                print_tree(child, indent + 1, visited)
        
        for root in sorted(roots):
            print_tree(root)
    
    # Display instance relationships
    print_info("📄 INSTANCES")
    if not ontology_manager.instance_of_relations:
        print("  No instances found in ontology")
    else:
        for instance, classes in sorted(ontology_manager.instance_of_relations.items()):
            class_list = ", ".join(sorted(classes))
            print(f"  {instance} → is instance of → {class_list}")
    
    # Display properties  
    print_info("🔑 PROPERTIES")
    if not ontology_manager.all_properties:
        print("  No properties found in ontology")
    else:
        for prop in sorted(ontology_manager.all_properties):
            print(f"  {prop}")
            
            # Show which entities have this property
            entities_with_prop = []
            for entity, props in ontology_manager.property_inheritance.items():
                if prop in props and props[prop]:
                    entities_with_prop.append(entity)
            
            if entities_with_prop:
                entities_str = ", ".join(sorted(entities_with_prop))
                print(f"    Entities with this property: {entities_str}")
            
            # Show exceptions
            exceptions = []
            for entity, props in ontology_manager.property_exceptions.items():
                if prop in props and not props[prop]:
                    exceptions.append(entity)
            
            if exceptions:
                exceptions_str = ", ".join(sorted(exceptions))
                print(f"    Exceptions (entities without this property): {exceptions_str}")

def display_extracted_rules(rules):
    print_header("EXTRACTED LOGICAL RULES")
    
    if not rules:
        print_warning("  No rules were extracted")
        return
    
    for i, rule in enumerate(rules, 1):
        rule_type = rule.get('type', 'unknown').upper()
        
        if rule_type == 'UNIVERSAL':
            statement = rule.get('statement', '')
            print(f"  {i}. {Colors.GREEN}UNIVERSAL:{Colors.ENDC} {statement}")
            
        elif rule_type == 'EXISTENTIAL':
            statement = rule.get('statement', '')
            print(f"  {i}. {Colors.BLUE}EXISTENTIAL:{Colors.ENDC} {statement}")
            
        elif rule_type == 'ASSERTION':
            statement = rule.get('statement', '')
            print(f"  {i}. {Colors.YELLOW}ASSERTION:{Colors.ENDC} {statement}")
            
        elif rule_type == 'NEGATION':
            statement = rule.get('statement', '')
            original = rule.get('original_text', '')
            print(f"  {i}. {Colors.RED}NEGATION:{Colors.ENDC} {original} (positive form: {statement})")
            
        elif rule_type == 'IMPLICATION':
            antecedent = rule.get('antecedent', '')
            consequent = rule.get('consequent', '')
            print(f"  {i}. {Colors.PURPLE}IMPLICATION:{Colors.ENDC} If {antecedent} then {consequent}")
            
        else:
            print(f"  {i}. {rule_type}: {rule}")

def verify_text(text, use_enhanced=True, show_details=True):
    print_header("VERIFICATION PROCESS")
    print_info("Verifying logical consistency...")
    
    flow_capture = InternalFlowCapture(debug=show_details)
    
    start_time = time.time()
    
    if use_enhanced:
        flow_capture.start_capture()
        verifier = EnhancedConsistencyVerifier()
        print_info("Using Enhanced Consistency Verifier")
        
        # Get extracted rules before verification
        # Create a new rule extractor since we can't access the one inside the verifier
        advanced_extractor = AdvancedRuleExtractor()
        extracted_rules = advanced_extractor.extract_rules(text)
        
        # Now verify
        result = verifier.verify(text)
        flow_capture.stop_capture()
        
        # Display internal details
        if show_details:
            display_extracted_rules(extracted_rules)
            display_internal_flow(flow_capture, detailed=show_details)
            display_ontology(verifier)
    else:
        verifier = ConsistencyVerifier()
        print_info("Using Standard Consistency Verifier")
        result = verifier.verify(text)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print_header("RESULTS")
    
    if result.is_consistent:
        print_success(f"✓ CONSISTENT (verified in {duration:.2f} seconds)")
    else:
        print_error(f"✗ INCONSISTENT (verified in {duration:.2f} seconds)")
        
        print_header("INCONSISTENCIES")
        for i, inconsistency in enumerate(result.inconsistencies, 1):
            print_error(f"{i}. {inconsistency}")
    
    return result, verifier

def repair_text(text, verifier):
    print_header("REPAIR PROCESS")
    print_info("Repairing inconsistencies...")
    
    start_time = time.time()
    repaired_text = verifier.repair(text)
    end_time = time.time()
    
    print_info(f"Repair completed in {end_time - start_time:.2f} seconds")
    
    print_header("REPAIRED TEXT")
    print(repaired_text)
    
    # Also verify the repaired text to ensure it's consistent
    print_header("VERIFICATION OF REPAIRED TEXT")
    print_info("Verifying that the repaired text is now consistent...")
    
    if isinstance(verifier, EnhancedConsistencyVerifier):
        result, _ = verify_text(repaired_text, use_enhanced=True, show_details=False)
    else:
        result, _ = verify_text(repaired_text, use_enhanced=False, show_details=False)
    
    return repaired_text

def compare_verifiers(text, show_details=True):
    print_header("COMPARISON MODE")
    print_info("Running both verifiers for comparison...")
    
    # Run standard verifier
    print_header("STANDARD VERIFIER")
    standard_verifier = ConsistencyVerifier()
    start_time = time.time()
    standard_result = standard_verifier.verify(text)
    standard_time = time.time() - start_time
    
    if standard_result.is_consistent:
        print_success(f"✓ CONSISTENT (verified in {standard_time:.2f} seconds)")
    else:
        print_error(f"✗ INCONSISTENT (verified in {standard_time:.2f} seconds)")
        print_header("STANDARD VERIFIER INCONSISTENCIES")
        for i, inconsistency in enumerate(standard_result.inconsistencies, 1):
            print_error(f"{i}. {inconsistency}")
    
    # Run enhanced verifier with internal flow capture
    print_header("ENHANCED VERIFIER")
    flow_capture = InternalFlowCapture(debug=show_details)
    flow_capture.start_capture()
    
    enhanced_verifier = EnhancedConsistencyVerifier()
    
    # Get extracted rules first
    if show_details:
        # Create a new rule extractor
        advanced_extractor = AdvancedRuleExtractor() 
        extracted_rules = advanced_extractor.extract_rules(text)
    
    start_time = time.time()
    enhanced_result = enhanced_verifier.verify(text)
    enhanced_time = time.time() - start_time
    
    flow_capture.stop_capture()
    
    if enhanced_result.is_consistent:
        print_success(f"✓ CONSISTENT (verified in {enhanced_time:.2f} seconds)")
    else:
        print_error(f"✗ INCONSISTENT (verified in {enhanced_time:.2f} seconds)")
        print_header("ENHANCED VERIFIER INCONSISTENCIES")
        for i, inconsistency in enumerate(enhanced_result.inconsistencies, 1):
            print_error(f"{i}. {inconsistency}")
    
    # Display enhanced verifier internals
    if show_details:
        display_extracted_rules(extracted_rules)
        display_internal_flow(flow_capture, detailed=show_details)
        display_ontology(enhanced_verifier)
    
    # Compare results
    print_header("COMPARISON SUMMARY")
    if standard_result.is_consistent == enhanced_result.is_consistent:
        print_success("Both verifiers agree on the consistency status")
    else:
        print_warning("Verifiers disagree on the consistency status")
    
    print_info(f"Standard verifier: {standard_time:.2f} seconds")
    print_info(f"Enhanced verifier: {enhanced_time:.2f} seconds")
    print_info(f"Speedup/slowdown: {standard_time/enhanced_time:.2f}x")
    
    return standard_result, enhanced_result

def show_menu():
    print_header("LLM CONSISTENCY VERIFIER")
    print("1. Verify text with Enhanced Verifier (detailed)")
    print("2. Verify text with Standard Verifier")
    print("3. Compare both verifiers (detailed)")
    print("4. Load example text")
    print("5. Toggle detail level (currently: " + ("DETAILED" if display_details else "SIMPLE") + ")")
    print("6. Exit")
    
    choice = input("\nEnter your choice (1-6): ")
    return choice

def load_example():
    examples = {
        "1": "All birds have feathers. Penguins are birds. Penguins have feathers.",
        "2": "All birds can fly. Penguins are birds. Penguins cannot fly.",
        "3": """All animals are living things.
All mammals are animals.
All cats are mammals.
Fluffy is a cat.
Fluffy is not a living thing.""",
        "4": """If it rains, then the ground gets wet.
If the ground is wet, then the grass is slippery.
If the grass is slippery, then it is dangerous to run.
It rained yesterday.
It was safe to run on the grass yesterday.""",
        "5": """All triangles have three sides.
All equilateral triangles are triangles.
All equilateral triangles have equal sides.
Shape X is not a triangle.
Shape X has three sides.
"""
    }
    
    print_header("EXAMPLE TEXTS")
    print("1. Consistent birds-feathers example")
    print("2. Inconsistent birds-fly example")
    print("3. Hierarchical inheritance example (Fluffy cat)")
    print("4. Complex conditional chain example")
    print("5. Contradiction with properties example")
    
    choice = input("\nSelect an example (1-5): ")
    
    if choice in examples:
        return examples[choice]
    else:
        print_error("Invalid choice. Using default example.")
        return examples["1"]

def main():
    global display_details
    display_details = True  # Default to showing details
    
    # Check for OpenAI API key
    if "OPENAI_API_KEY" not in os.environ:
        print_error("ERROR: OPENAI_API_KEY environment variable not found!")
        print_info("Please set your OpenAI API key before running this script:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    while True:
        choice = show_menu()
        
        if choice == "1":
            text = get_multiline_input()
            result, verifier = verify_text(text, use_enhanced=True, show_details=display_details)
            
            if not result.is_consistent:
                repair_choice = input("\nWould you like to repair the inconsistencies? (y/n): ").lower()
                if repair_choice == 'y':
                    repair_text(text, verifier)
        
        elif choice == "2":
            text = get_multiline_input()
            result, verifier = verify_text(text, use_enhanced=False, show_details=False)
            
            if not result.is_consistent:
                repair_choice = input("\nWould you like to repair the inconsistencies? (y/n): ").lower()
                if repair_choice == 'y':
                    repair_text(text, verifier)
        
        elif choice == "3":
            text = get_multiline_input()
            standard_result, enhanced_result = compare_verifiers(text, show_details=display_details)
            
            if not enhanced_result.is_consistent:
                repair_choice = input("\nWould you like to repair the inconsistencies? (y/n): ").lower()
                if repair_choice == 'y':
                    verifier = EnhancedConsistencyVerifier()
                    repair_text(text, verifier)
        
        elif choice == "4":
            text = load_example()
            print_header("LOADED EXAMPLE")
            print(text)
            
            choice = input("\nWhich verifier would you like to use? (1=Enhanced, 2=Standard, 3=Compare): ")
            
            if choice == "1":
                result, verifier = verify_text(text, use_enhanced=True, show_details=display_details)
                if not result.is_consistent:
                    repair_choice = input("\nWould you like to repair the inconsistencies? (y/n): ").lower()
                    if repair_choice == 'y':
                        repair_text(text, verifier)
            
            elif choice == "2":
                result, verifier = verify_text(text, use_enhanced=False, show_details=False)
                if not result.is_consistent:
                    repair_choice = input("\nWould you like to repair the inconsistencies? (y/n): ").lower()
                    if repair_choice == 'y':
                        repair_text(text, verifier)
            
            elif choice == "3":
                standard_result, enhanced_result = compare_verifiers(text, show_details=display_details)
                if not enhanced_result.is_consistent:
                    repair_choice = input("\nWould you like to repair the inconsistencies? (y/n): ").lower()
                    if repair_choice == 'y':
                        verifier = EnhancedConsistencyVerifier()
                        repair_text(text, verifier)

        elif choice == "5":
            display_details = not display_details
            print_info(f"Detail level set to: {display_details}")
        
        elif choice == "6":
            print_info("Thank you for using the LLM Consistency Verifier!")
            break
        
        else:
            print_error("Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main() 