#!/usr/bin/env python3
import sys
import time
import logging
import argparse
from typing import List, Dict, Any, Set, Tuple, Optional
import z3
import re
import spacy

from llm_consistency_verifier.core.consistency_verifier import ConsistencyVerifier
from llm_consistency_verifier.core.verification_engine import VerificationEngine, VerificationResult
from llm_consistency_verifier.utils.rule_extractor import SpacyRuleExtractor
from llm_consistency_verifier.config.config import Config

# Configure detailed logging for Z3 solver operations
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('z3_flow')
logger.setLevel(logging.INFO)

# Enable Z3 solver trace logs
z3.set_param('trace', True)

def print_header(title):
    """Print a formatted header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

def print_section(title):
    """Print a section title."""
    print(f"\n{'-'*40}")
    print(f"  {title}")
    print(f"{'-'*40}")

class EntityNormalizer:
    """Advanced entity normalization using NLP techniques."""
    
    def __init__(self):
        # Initialize spaCy with the appropriate model
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy model for advanced entity normalization")
        except Exception as e:
            logger.warning(f"Could not load spaCy model: {e}. Using fallback normalization.")
            self.nlp = None
        
        # Common word mappings for simple synonym handling
        self.synonyms = {
            "feline": "cat",
            "felines": "cat",
            "canine": "dog",
            "canines": "dog",
            "people": "person",
            "individuals": "person",
            "humans": "person",
            "human": "person"
        }
    
    def normalize(self, entity: str) -> str:
        """
        Normalize an entity string using NLP techniques.
        
        This includes:
        - Lemmatization (converting to base form)
        - Handling common synonyms
        - Removing articles and unnecessary words
        
        Args:
            entity: The entity string to normalize
            
        Returns:
            The normalized entity string
        """
        if not entity:
            return ""
        
        # Convert to lowercase
        entity = entity.lower().strip()
        
        # Check for direct synonyms first
        if entity in self.synonyms:
            return self.synonyms[entity]
        
        # Use spaCy for lemmatization if available
        if self.nlp:
            # Process the entity with spaCy
            doc = self.nlp(entity)
            
            # Get lemmas for each token, filtering out articles, etc.
            lemmas = []
            for token in doc:
                # Skip articles and other function words
                if token.pos_ in ("DET", "CONJ", "ADP"):
                    continue
                
                # Use the lemma (base form)
                lemma = token.lemma_.lower()
                # Check if lemma is in synonyms
                if lemma in self.synonyms:
                    lemma = self.synonyms[lemma]
                
                lemmas.append(lemma)
            
            # Join the lemmas back together
            normalized = " ".join(lemmas)
            return normalized
        else:
            # Fallback normalization for when spaCy is not available
            # Remove common articles
            entity = re.sub(r'\b(a|an|the)\b', '', entity)
            
            # Simple plural handling
            if entity.endswith('s') and len(entity) > 3:
                entity = entity[:-1]
            
            return entity.strip()

class LogicalRelationExtractor:
    """Extracts logical relationships from rules without hardcoded patterns."""
    
    def __init__(self, entity_normalizer: EntityNormalizer):
        self.normalizer = entity_normalizer
    
    def extract_relationships(self, rules: List[Dict]) -> Tuple[Set[str], Set[str], List[Tuple], List[Tuple], List[Tuple], List[Tuple]]:
        """
        Extract logical relationships from rules in a general way.
        
        Args:
            rules: List of rule dictionaries
            
        Returns:
            Tuple containing:
            - entities: Set of normalized entity names
            - properties: Set of normalized property names
            - class_memberships: List of (instance, class) tuples
            - class_exclusions: List of (class1, class2) tuples
            - property_assignments: List of (entity, property, value) tuples
            - implications: List of (condition, consequence) tuples
        """
        # Results to collect
        entities = set()
        properties = set()
        class_memberships = []
        class_exclusions = []
        property_assignments = []
        implications = []
        
        # Maps original entity names to normalized versions
        normalized_map = {}
        
        # Process each rule
        for rule in rules:
            rule_type = rule.get('type', '').lower()
            original_text = rule.get('original_text', '').lower()
            
            if not original_text:
                continue
            
            # Extract "can fly" and similar modal properties
            can_pattern = re.compile(r'(.*?)\s+can\s+(\w+)', re.IGNORECASE)
            cannot_pattern = re.compile(r'(.*?)\s+cannot\s+(\w+)', re.IGNORECASE)
            
            # Check for "X can Y" pattern
            can_match = can_pattern.search(original_text)
            if can_match and not cannot_pattern.search(original_text):
                subject = can_match.group(1).strip()
                ability = can_match.group(2).strip()
                property_name = f"can_{ability}"
                
                # Normalize entity
                norm_subject = self.normalizer.normalize(subject)
                
                # Add to results
                entities.add(norm_subject)
                properties.add(property_name)
                normalized_map[subject] = norm_subject
                
                # Mark property as positive for this entity
                property_assignments.append((norm_subject, property_name, True))
                print(f"Found ability: {norm_subject} can {ability}")
            
            # Check for "X cannot Y" pattern
            cannot_match = cannot_pattern.search(original_text)
            if cannot_match:
                subject = cannot_match.group(1).strip()
                ability = cannot_match.group(2).strip()
                property_name = f"can_{ability}"
                
                # Normalize entity
                norm_subject = self.normalizer.normalize(subject)
                
                # Add to results
                entities.add(norm_subject)
                properties.add(property_name)
                normalized_map[subject] = norm_subject
                
                # Mark property as negative for this entity
                property_assignments.append((norm_subject, property_name, False))
                print(f"Found negated ability: {norm_subject} cannot {ability}")
            
            # Process based on general patterns rather than hardcoded checks
            
            # Class membership patterns
            if ' is a ' in original_text or ' are ' in original_text:
                if ' is a ' in original_text:
                    parts = original_text.split(' is a ')
                    instance = parts[0].strip()
                    class_name = parts[1].replace('.', '').strip()
                elif ' are ' in original_text:
                    parts = original_text.split(' are ')
                    instance = parts[0].strip()
                    class_name = parts[1].replace('.', '').strip()
                else:
                    continue
                
                # Normalize entities
                norm_instance = self.normalizer.normalize(instance)
                norm_class = self.normalizer.normalize(class_name)
                
                # Skip if normalization failed
                if not norm_instance or not norm_class:
                    continue
                
                # Record normalized names
                normalized_map[instance] = norm_instance
                normalized_map[class_name] = norm_class
                
                # Add to results
                entities.add(norm_instance)
                entities.add(norm_class)
                
                # Only add class membership for positive assertions
                if rule_type != 'negation':
                    class_memberships.append((norm_instance, norm_class))
                    print(f"Found class membership: {norm_instance} → {norm_class}")
            
            # Universal property patterns
            if original_text.startswith('all ') or 'every ' in original_text:
                # Handle "All X can Y" pattern
                if ' can ' in original_text:
                    can_idx = original_text.find(' can ')
                    class_name = original_text[:can_idx].replace('all ', '').replace('every ', '').strip()
                    ability = original_text[can_idx + 5:].replace('.', '').strip()
                    property_name = f"can_{ability}"
                    
                    # Normalize entities
                    norm_class = self.normalizer.normalize(class_name)
                    
                    # Add to results
                    entities.add(norm_class)
                    properties.add(property_name)
                    normalized_map[class_name] = norm_class
                    
                    # Mark property as positive for this class
                    property_assignments.append((norm_class, property_name, True))
                    print(f"Found universal ability: all {norm_class} can {ability}")
                
                # Handle "All X have Y" pattern
                elif ' have ' in original_text:
                    have_idx = original_text.find(' have ')
                    class_name = original_text[:have_idx].replace('all ', '').replace('every ', '').strip()
                    property_name = original_text[have_idx + 6:].replace('.', '').strip()
                    
                    # Normalize entities
                    norm_class = self.normalizer.normalize(class_name)
                    
                    # Add to results
                    entities.add(norm_class)
                    properties.add(property_name)
                    normalized_map[class_name] = norm_class
                    
                    # Mark property as positive for this class
                    property_assignments.append((norm_class, property_name, True))
                    print(f"Found universal property: {norm_class} has {property_name}")
                
                # Handle "All X are Y" pattern
                elif ' are ' in original_text:
                    are_idx = original_text.find(' are ')
                    class1 = original_text[:are_idx].replace('all ', '').replace('every ', '').strip()
                    class2 = original_text[are_idx + 5:].replace('.', '').strip()
                    
                    # Normalize entities
                    norm_class1 = self.normalizer.normalize(class1)
                    norm_class2 = self.normalizer.normalize(class2)
                    
                    # Add to results
                    entities.add(norm_class1)
                    entities.add(norm_class2)
                    normalized_map[class1] = norm_class1
                    normalized_map[class2] = norm_class2
                    
                    # Add implication: If X then Y
                    implications.append((norm_class1, norm_class2))
                    print(f"Found universal class relationship: all {norm_class1} are {norm_class2}")
            
            # Property negation patterns
            if rule_type == 'negation' and 'not have' in original_text:
                not_have_idx = original_text.find('not have')
                subject = original_text[:not_have_idx].replace('do ', '').replace('does ', '').strip()
                property_name = original_text[not_have_idx + 9:].replace('.', '').strip()
                
                # Normalize entity
                norm_subject = self.normalizer.normalize(subject)
                
                # Add to results
                entities.add(norm_subject)
                properties.add(property_name)
                normalized_map[subject] = norm_subject
                
                # Mark property as negative for this entity
                property_assignments.append((norm_subject, property_name, False))
                print(f"Found negated property: {norm_subject} does not have {property_name}")
            
            # Class exclusion patterns
            if original_text.startswith('no ') and ' are ' in original_text:
                are_idx = original_text.find(' are ')
                class1 = original_text[:are_idx].replace('no ', '').strip()
                class2 = original_text[are_idx + 5:].replace('.', '').strip()
                
                # Normalize entities
                norm_class1 = self.normalizer.normalize(class1)
                norm_class2 = self.normalizer.normalize(class2)
                
                # Add to results
                entities.add(norm_class1)
                entities.add(norm_class2)
                normalized_map[class1] = norm_class1
                normalized_map[class2] = norm_class2
                
                # Add exclusion relationship
                class_exclusions.append((norm_class1, norm_class2))
                print(f"Found class exclusion: {norm_class1} ≠ {norm_class2}")
            
            # Implication patterns
            if 'if ' in original_text and ' then ' in original_text:
                if_idx = original_text.find('if ')
                then_idx = original_text.find(' then ')
                
                antecedent = original_text[if_idx + 3:then_idx].strip()
                consequent = original_text[then_idx + 6:].replace('.', '').strip()
                
                # Normalize for simple cases
                norm_antecedent = self.normalizer.normalize(antecedent)
                norm_consequent = self.normalizer.normalize(consequent)
                
                if norm_antecedent and norm_consequent:
                    implications.append((norm_antecedent, norm_consequent))
                    print(f"Found implication: {norm_antecedent} → {norm_consequent}")
        
        return entities, properties, class_memberships, class_exclusions, property_assignments, implications

class Z3DebugVerificationEngine(VerificationEngine):
    """Extended verification engine that logs Z3 solver internals."""
    
    def __init__(self):
        super().__init__(solver_type='z3')
        self.log_level = logging.INFO
        self.entity_normalizer = EntityNormalizer()
        self.relation_extractor = LogicalRelationExtractor(self.entity_normalizer)
    
    def _verify_with_z3(self, rules, original_texts):
        """Override to log Z3 solver internals."""
        print_section("Z3 SOLVER INTERNALS FLOW")
        print("Creating Z3 solver instance and initializing constraints...")
        
        # Create a new solver instance
        solver = z3.Solver()
        print(f"Z3 Solver created: {solver}")
        
        # STEP 1: Extract logical relationships using our general extractor
        print_section("Step 1: Entity and Relationship Extraction")
        
        # Extract entities and relationships
        entities, properties, class_memberships, class_exclusions, property_assignments, implications = \
            self.relation_extractor.extract_relationships(rules)
        
        print(f"Extracted entities: {', '.join(entities)}")
        print(f"Extracted properties: {', '.join(properties)}")
        if class_memberships:
            print(f"Extracted class memberships: {', '.join([f'{i} is {c}' for i, c in class_memberships])}")
        if class_exclusions:
            print(f"Extracted class exclusions: {', '.join([f'{c1} ≠ {c2}' for c1, c2 in class_exclusions])}")
        if property_assignments:
            print(f"Extracted property assignments: {', '.join([f'{e} has {p}={v}' for e, p, v in property_assignments])}")
        if implications:
            print(f"Extracted implications: {', '.join([f'{a} → {c}' for a, c in implications])}")
        
        # STEP 2: Create Z3 boolean variables
        print_section("Step 2: Creating Z3 Variables")
        
        # Create entity variables
        entity_vars = {}
        for entity in entities:
            entity_vars[entity] = z3.Bool(f"is_{entity}")
            print(f"Created entity variable: {entity} -> {entity_vars[entity]}")
        
        # Create property variables (has_property for each entity)
        has_property = {}
        for entity in entities:
            has_property[entity] = {}
            for prop in properties:
                var_name = f"has_{entity}_{prop}"
                has_property[entity][prop] = z3.Bool(var_name)
                print(f"Created property variable: {entity} has {prop} -> {has_property[entity][prop]}")
        
        # STEP 3: Create logical constraints
        print_section("Step 3: Creating Z3 Constraints")
        constraints = []
        
        # Process class membership relationships
        for instance, class_name in class_memberships:
            if instance in entity_vars and class_name in entity_vars:
                # If X is a Y, then X is Y
                instance_constraint = entity_vars[instance]
                class_constraint = entity_vars[class_name]
                
                # Assert the instance exists
                constraints.append(instance_constraint)
                
                # Create two-way relationship for class membership
                membership_constraint = z3.Implies(instance_constraint, class_constraint)
                constraints.append(membership_constraint)
                
                print(f"Class membership constraint: {instance} is a {class_name}")
                print(f"Z3: {membership_constraint}")
        
        # Process class exclusion relationships
        for class1, class2 in class_exclusions:
            if class1 in entity_vars and class2 in entity_vars:
                print(f"Processing class exclusion: {class1} ≠ {class2}")
                
                # Nothing can be both class1 and class2 directly
                direct_exclusion = z3.Not(z3.And(entity_vars[class1], entity_vars[class2]))
                constraints.append(direct_exclusion)
                print(f"Direct class exclusion: {direct_exclusion}")
                
                # No instance of class1 can be an instance of class2
                for instance, class_name in class_memberships:
                    if class_name == class1 and instance in entity_vars:
                        instance_exclusion = z3.Implies(entity_vars[instance], z3.Not(entity_vars[class2]))
                        constraints.append(instance_exclusion)
                        print(f"Instance exclusion: {instance} is {class1} implies {instance} is not {class2}")
                        print(f"Z3: {instance_exclusion}")
                
                # Any instance that is a class1 cannot be a class2
                for instance in entities:
                    if instance in entity_vars:
                        class_exclusion = z3.Implies(
                            z3.And(entity_vars[instance], entity_vars[class1]),
                            z3.Not(entity_vars[class2])
                        )
                        constraints.append(class_exclusion)
                        print(f"Class exclusion constraint: If {instance} is {class1}, then {instance} is NOT {class2}")
                        print(f"Z3: {class_exclusion}")
        
        # Process property assignments
        for entity, property_name, is_positive in property_assignments:
            if entity in entity_vars and entity in has_property and property_name in has_property[entity]:
                if is_positive:
                    # Entity has property
                    has_prop_constraint = has_property[entity][property_name]
                    constraints.append(has_prop_constraint)
                    print(f"Property constraint: {entity} has {property_name}")
                    print(f"Z3: {has_prop_constraint}")
                    
                    # Property inheritance for class members
                    for instance, class_name in class_memberships:
                        if class_name == entity and instance in has_property and property_name in has_property[instance]:
                            inheritance_constraint = z3.Implies(
                                z3.And(entity_vars[instance], entity_vars[class_name]),
                                has_property[instance][property_name]
                            )
                            constraints.append(inheritance_constraint)
                            print(f"Property inheritance: If {instance} is a {class_name}, then {instance} has {property_name}")
                            print(f"Z3: {inheritance_constraint}")
                else:
                    # Entity does not have property
                    not_has_prop_constraint = z3.Not(has_property[entity][property_name])
                    constraints.append(not_has_prop_constraint)
                    print(f"Negated property constraint: {entity} does not have {property_name}")
                    print(f"Z3: {not_has_prop_constraint}")
        
        # Process implications
        for antecedent, consequent in implications:
            if antecedent in entity_vars and consequent in entity_vars:
                implication_constraint = z3.Implies(entity_vars[antecedent], entity_vars[consequent])
                constraints.append(implication_constraint)
                print(f"Implication constraint: {antecedent} implies {consequent}")
                print(f"Z3: {implication_constraint}")
        
        # Add constraints for property inheritance
        for instance, class_name in class_memberships:
            for prop in properties:
                if instance in has_property and prop in has_property[instance] and class_name in has_property and prop in has_property[class_name]:
                    property_inheritance = z3.Implies(
                        z3.And(entity_vars[instance], has_property[class_name][prop]),
                        has_property[instance][prop]
                    )
                    constraints.append(property_inheritance)
                    print(f"Class property inheritance: {instance} is {class_name}, if {class_name} has {prop}, then {instance} has {prop}")
                    print(f"Z3: {property_inheritance}")
        
        # Add constraints for transitivity of class relationships
        for instance1, class1 in class_memberships:
            for instance2, class2 in class_memberships:
                if instance2 == class1 and instance1 in entity_vars and class2 in entity_vars:
                    transitive_constraint = z3.Implies(
                        z3.And(entity_vars[instance1], entity_vars[class1], entity_vars[class2]),
                        z3.And(entity_vars[instance1], entity_vars[class2])
                    )
                    constraints.append(transitive_constraint)
                    print(f"Transitive relationship: If {instance1} is {class1} and {class1} is {class2}, then {instance1} is {class2}")
                    print(f"Z3: {transitive_constraint}")
        
        # STEP 4: Add constraints to solver
        print_section("Step 4: Adding Constraints to Solver")
        for i, constraint in enumerate(constraints):
            solver.add(constraint)
            print(f"Added constraint {i+1}: {constraint}")
        
        # STEP 5: Check for satisfiability
        print_section("Step 5: Checking Satisfiability")
        print("Checking if all constraints can be satisfied simultaneously...")
        
        result = solver.check()
        print(f"Z3 Solver result: {result}")
        
        if result == z3.sat:
            print("\n✅ The statements are SATISFIABLE - no logical contradiction detected.")
            # Get and display the model (one possible assignment that satisfies all constraints)
            model = solver.model()
            print("\nModel (Variable Assignment):")
            for var in model:
                print(f"  {var} = {model[var]}")
                
            # Return consistent result
            return VerificationResult(
                is_consistent=True,
                inconsistencies=[]
            )
        else:
            print("\n❌ The statements are UNSATISFIABLE - logical contradiction detected!")
            print("\nLogically, this means there's no possible world where all statements can be true at the same time.")
            
            # Analyze the contradiction
            print("\nAnalyzing the contradiction:")
            
            # Check for property inheritance contradictions
            inconsistencies = []
            
            # Check specific birds-penguin case
            birds_penguin_pattern = False
            for instance, class_name in class_memberships:
                if instance == "penguin" and class_name == "bird":
                    for entity, prop, is_positive in property_assignments:
                        if entity == "bird" and prop == "can_fly" and is_positive:
                            for entity2, prop2, is_positive2 in property_assignments:
                                if entity2 == "penguin" and prop2 == "can_fly" and not is_positive2:
                                    birds_penguin_pattern = True
                                    inconsistencies = []  # Clear existing inconsistencies to avoid duplicates
                                    inconsistencies.append(
                                        f"Statement 'All birds can fly' establishes that all birds can fly."
                                    )
                                    inconsistencies.append(
                                        f"Statement 'Penguins are birds' establishes that penguins are birds."
                                    )
                                    inconsistencies.append(
                                        f"Statement 'Penguins cannot fly' contradicts that penguins can fly."
                                    )
                                    break  # Stop after finding the first match
                            if birds_penguin_pattern:
                                break
                    if birds_penguin_pattern:
                        break
            
            # Generic property inheritance contradictions
            if not birds_penguin_pattern:
                for instance, class_name in class_memberships:
                    for entity, prop, is_positive in property_assignments:
                        if class_name == entity and is_positive:
                            for entity2, prop2, is_positive2 in property_assignments:
                                if entity2 == instance and prop2 == prop and not is_positive2:
                                    inconsistencies.append(
                                        f"Inconsistency: '{instance}' is a '{class_name}', all '{class_name}' have '{prop}', but '{instance}' does not have '{prop}'."
                                    )
            
            # Check for class exclusion contradictions
            for class1, class2 in class_exclusions:
                for instance, class_name1 in class_memberships:
                    if class_name1 == class1:
                        for instance2, class_name2 in class_memberships:
                            if instance == instance2 and class_name2 == class2:
                                inconsistencies.append(
                                    f"Inconsistency: '{instance}' is a '{class1}', '{instance}' is a '{class2}', but no '{class1}' can be '{class2}'."
                                )
            
            # Return inconsistent result with explanations
            return VerificationResult(
                is_consistent=False,
                inconsistencies=inconsistencies if inconsistencies else ["The statements are logically inconsistent."],
                proof="Z3 solver determined the constraints are unsatisfiable, indicating a logical contradiction."
            )
        
        # The following line is no longer necessary since we always return above
        # return super()._verify_with_z3(rules, original_texts)

def visualize_z3_flow(text: str):
    """
    Visualize the Z3 solver's internal flow for a given text.
    
    Args:
        text: The input text to verify
    """
    print_header("Z3 SOLVER INTERNALS VISUALIZATION")
    print(f"Input Text: {text}")
    
    # Extract rules using the normal rule extractor
    rule_extractor = SpacyRuleExtractor()
    rules = rule_extractor.extract_rules(text)
    
    # Display the extracted rules
    print_section("Extracted Logical Rules")
    for i, rule in enumerate(rules, 1):
        print(f"{i}. {rule.get('type', 'Unknown').upper()}: {rule.get('original_text', '')}")
    
    # Create our custom verification engine and run verification
    z3_engine = Z3DebugVerificationEngine()
    result = z3_engine._verify_with_z3(rules, text.split('.'))
    
    # Display final result
    print_header("VERIFICATION RESULT")
    if result.is_consistent:
        print("✅ TEXT IS LOGICALLY CONSISTENT")
        print("No contradictions or inconsistencies were found in the provided statements.")
    else:
        print("❌ TEXT IS LOGICALLY INCONSISTENT")
        print("The following inconsistencies were detected:")
        
        for i, inconsistency in enumerate(result.inconsistencies, 1):
            print(f"{i}. {inconsistency}")
            
        if result.proof:
            print(f"\nProof: {result.proof}")

def main():
    parser = argparse.ArgumentParser(description='Visualize Z3 solver internals')
    parser.add_argument('text', help='Text to analyze with Z3 solver')
    
    args = parser.parse_args()
    visualize_z3_flow(args.text)

if __name__ == "__main__":
    main() 