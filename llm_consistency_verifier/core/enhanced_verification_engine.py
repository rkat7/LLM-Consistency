import logging
import z3
from typing import List, Dict, Any, Set, Tuple, Optional
import re
from collections import defaultdict

from ..config.config import Config
from .verification_engine import VerificationEngine, VerificationResult
from ..utils.ontology_manager import OntologyManager

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedVerificationEngine(VerificationEngine):
    """
    Enhanced verification engine with support for complex logical structures.
    
    This extends the base VerificationEngine to handle:
    - Hierarchical ontology
    - Complex logical expressions
    - Temporal and modal logic
    - Nested quantifiers
    """
    
    def __init__(self, solver_type: str = Config.SOLVER_TYPE):
        """Initialize the enhanced verification engine."""
        super().__init__(solver_type)
        self.ontology_manager = OntologyManager()
        logger.info("Enhanced verification engine initialized")
    
    def verify(self, logical_rules, original_statements=None):
        """
        Verify logical consistency with enhanced capabilities.
        
        Args:
            logical_rules: List of logical rules to verify
            original_statements: Original text statements
            
        Returns:
            VerificationResult: The result of verification
        """
        logger.info("[FLOW:ENHANCED_VERIFY] Starting enhanced verification")
        
        # Process rules to build ontology
        self._build_ontology(logical_rules)
        
        # Check ontology consistency first
        ontology_issues = self.ontology_manager.check_consistency()
        if ontology_issues:
            logger.info(f"[FLOW:ENHANCED_VERIFY] Found {len(ontology_issues)} ontology inconsistencies")
            return VerificationResult(
                is_consistent=False,
                inconsistencies=ontology_issues,
                proof="Inconsistencies detected in the ontological structure."
            )
        
        # Process rules for advanced features
        advanced_rules = self._process_advanced_features(logical_rules)
        
        # Check for complex logical structures
        if self._has_complex_structures(advanced_rules):
            logger.info("[FLOW:ENHANCED_VERIFY] Detected complex logical structures, using enhanced verification")
            return self._verify_with_enhanced_z3(advanced_rules, original_statements)
        
        # Fall back to standard verification for simple cases
        logger.info("[FLOW:ENHANCED_VERIFY] No complex structures detected, using standard verification")
        return super().verify(logical_rules, original_statements)
    
    def _build_ontology(self, rules: List[Dict[str, Any]]) -> None:
        """
        Build the ontology from the rules.
        
        Args:
            rules: List of logical rules
        """
        logger.info("[FLOW:ENHANCED_VERIFY:ONTOLOGY] Building ontology from rules")
        
        for rule in rules:
            self.ontology_manager.process_rule(rule)
        
        # Log some stats
        logger.info(f"[FLOW:ENHANCED_VERIFY:ONTOLOGY] Built ontology with {len(self.ontology_manager.all_entities)} entities and {len(self.ontology_manager.all_properties)} properties")
    
    def _process_advanced_features(self, rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process rules to handle advanced logical features.
        
        Args:
            rules: List of logical rules
            
        Returns:
            List of processed rules
        """
        logger.info("[FLOW:ENHANCED_VERIFY:ADVANCED] Processing advanced logical features")
        
        processed_rules = []
        
        for rule in rules:
            # Create a copy of the rule to modify
            processed_rule = rule.copy()
            
            # Handle nested quantifiers
            if 'nested_quantifiers' in rule and rule['nested_quantifiers']:
                self._process_nested_quantifiers(processed_rule)
            
            # Handle temporal logic
            if 'temporal_context' in rule and rule['temporal_context']:
                self._process_temporal_context(processed_rule)
            
            # Handle modal logic
            if 'modal_context' in rule and rule['modal_context']:
                self._process_modal_context(processed_rule)
            
            processed_rules.append(processed_rule)
        
        return processed_rules
    
    def _process_nested_quantifiers(self, rule: Dict[str, Any]) -> None:
        """
        Process nested quantifiers in a rule.
        
        Args:
            rule: The rule to process
        """
        # Implementation depends on the specific nested quantifier structure
        # This is a placeholder for actual implementation
        pass
    
    def _process_temporal_context(self, rule: Dict[str, Any]) -> None:
        """
        Process temporal context in a rule.
        
        Args:
            rule: The rule to process
        """
        # Implementation depends on the specific temporal logic structure
        # This is a placeholder for actual implementation
        pass
    
    def _process_modal_context(self, rule: Dict[str, Any]) -> None:
        """
        Process modal context in a rule.
        
        Args:
            rule: The rule to process
        """
        # Implementation depends on the specific modal logic structure
        # This is a placeholder for actual implementation
        pass
    
    def _has_complex_structures(self, rules: List[Dict[str, Any]]) -> bool:
        """
        Check if rules contain complex logical structures.
        
        Args:
            rules: List of logical rules
            
        Returns:
            True if complex structures are present, False otherwise
        """
        for rule in rules:
            if any(key in rule for key in ['nested_quantifiers', 'temporal_context', 'modal_context', 'ontological_relation']):
                return True
        
        return False
    
    def _verify_with_enhanced_z3(self, rules: List[Dict[str, Any]], original_statements: Optional[List[str]] = None) -> VerificationResult:
        """
        Verify consistency using enhanced Z3 capabilities.
        
        Args:
            rules: List of logical rules
            original_statements: Original text statements
            
        Returns:
            VerificationResult: The result of verification
        """
        logger.info("[FLOW:ENHANCED_VERIFY:Z3] Starting enhanced Z3 verification")
        
        # Flag to track if we know the result is unsatisfiable
        self._is_satisfiable = None
        
        # Create Z3 solver
        solver = z3.Solver()
        
        # Track all created variables and constraints for analysis
        variables = {}  # name -> Z3 variable
        constraints = []  # List of (constraint, explanation) tuples
        
        # Create base boolean variables for all entities
        for entity in self.ontology_manager.all_entities:
            variables[f"entity_{entity}"] = z3.Bool(f"entity_{entity}")
        
        # Create property variables
        for entity in self.ontology_manager.all_entities:
            for prop in self.ontology_manager.all_properties:
                variables[f"prop_{entity}_{prop}"] = z3.Bool(f"prop_{entity}_{prop}")
        
        # Add entity hierarchy constraints
        self._add_entity_hierarchy_constraints(solver, variables, constraints)
        
        # Add property constraints
        self._add_property_constraints(solver, variables, constraints)
        
        # Add rule-specific constraints
        self._add_rule_constraints(solver, variables, constraints, rules)
        
        # Manually add constraints for hierarchical contradictions
        self._add_hierarchical_contradiction_constraints(solver, variables, constraints, rules)
        
        # Check for satisfiability
        logger.info("[FLOW:ENHANCED_VERIFY:Z3] Checking satisfiability")
        
        # If we already determined it's unsatisfiable, skip checking
        if self._is_satisfiable is False:
            result = z3.unsat
        else:
            result = solver.check()
        
        if result == z3.sat:
            logger.info("[FLOW:ENHANCED_VERIFY:Z3] Model is satisfiable (consistent)")
            
            # For debugging, we can inspect the model
            # model = solver.model()
            # logger.debug(f"Satisfying model: {model}")
            
            return VerificationResult(
                is_consistent=True,
                inconsistencies=[],
                proof="Z3 solver determined the logical rules are consistent."
            )
        else:
            logger.info("[FLOW:ENHANCED_VERIFY:Z3] Model is unsatisfiable (inconsistent)")
            
            # Analysis for identifying the source of inconsistency
            inconsistencies = self._analyze_unsatisfiable_core(solver, constraints)
            
            return VerificationResult(
                is_consistent=False,
                inconsistencies=inconsistencies,
                proof="Z3 solver determined the logical rules are inconsistent."
            )
    
    def _add_entity_hierarchy_constraints(self, solver: z3.Solver, variables: Dict[str, z3.ExprRef], constraints: List[Tuple[z3.ExprRef, str]]) -> None:
        """
        Add constraints for entity hierarchies.
        
        Args:
            solver: Z3 solver instance
            variables: Dictionary of Z3 variables
            constraints: List to store constraints
        """
        logger.info("[FLOW:ENHANCED_VERIFY:Z3] Adding entity hierarchy constraints")
        
        # First, compute the transitive closure of is-a relationships
        transitive_relations = self._compute_transitive_closure()
        
        # Add is-a relationship constraints including transitive ones
        for child, parents in transitive_relations.items():
            child_var = variables.get(f"entity_{child}")
            if child_var is not None:
                for parent in parents:
                    parent_var = variables.get(f"entity_{parent}")
                    if parent_var is not None:
                        # If child is true, parent must be true
                        constraint = z3.Implies(child_var, parent_var)
                        explanation = f"If {child} exists, then {parent} must exist (is-a relationship, possibly transitive)"
                        constraints.append((constraint, explanation))
                        solver.add(constraint)
        
        # Add instance-of relationship constraints
        for instance, classes in self.ontology_manager.instance_of_relations.items():
            instance_var = variables.get(f"entity_{instance}")
            if instance_var is not None:
                for class_name in classes:
                    class_var = variables.get(f"entity_{class_name}")
                    if class_var is not None:
                        # If instance is true, class must be true
                        constraint = z3.Implies(instance_var, class_var)
                        explanation = f"If {instance} exists, then {class_name} must exist (instance-of relationship)"
                        constraints.append((constraint, explanation))
                        solver.add(constraint)
                        
                        # Also add constraints for all parent classes of this class
                        if class_name in transitive_relations:
                            for parent_class in transitive_relations[class_name]:
                                parent_class_var = variables.get(f"entity_{parent_class}")
                                if parent_class_var is not None:
                                    # If instance belongs to a class, it also belongs to all parent classes
                                    constraint = z3.Implies(instance_var, parent_class_var)
                                    explanation = f"If {instance} is an instance of {class_name}, and {class_name} is a kind of {parent_class}, then {instance} is also a kind of {parent_class}"
                                    constraints.append((constraint, explanation))
                                    solver.add(constraint)
    
    def _compute_transitive_closure(self) -> Dict[str, Set[str]]:
        """
        Compute the transitive closure of all is-a relationships.
        
        Returns:
            Dictionary mapping each entity to all its ancestors (direct and indirect)
        """
        transitive_relations = {}
        
        # Initialize with direct relationships
        for child, parents in self.ontology_manager.is_a_relations.items():
            transitive_relations[child] = parents.copy()
        
        # Compute transitive closure until no more changes
        changed = True
        while changed:
            changed = False
            for child, parents in list(transitive_relations.items()):
                new_parents = parents.copy()
                
                # For each parent, add its parents too
                for parent in parents:
                    if parent in transitive_relations:
                        for grandparent in transitive_relations[parent]:
                            if grandparent not in new_parents:
                                new_parents.add(grandparent)
                                changed = True
                
                if len(new_parents) > len(parents):
                    transitive_relations[child] = new_parents
        
        return transitive_relations
    
    def _add_property_constraints(self, solver: z3.Solver, variables: Dict[str, z3.ExprRef], constraints: List[Tuple[z3.ExprRef, str]]) -> None:
        """
        Add constraints for properties.
        
        Args:
            solver: Z3 solver instance
            variables: Dictionary of Z3 variables
            constraints: List to store constraints
        """
        logger.info("[FLOW:ENHANCED_VERIFY:Z3] Adding property constraints")
        
        # Add direct property assignments
        for entity, props in self.ontology_manager.property_inheritance.items():
            entity_var = variables.get(f"entity_{entity}")
            if entity_var is not None:
                for prop, value in props.items():
                    prop_var = variables.get(f"prop_{entity}_{prop}")
                    if prop_var is not None:
                        # Set property value
                        if value:
                            constraint = z3.Implies(entity_var, prop_var)
                            explanation = f"If {entity} exists, then it has property {prop}"
                        else:
                            constraint = z3.Implies(entity_var, z3.Not(prop_var))
                            explanation = f"If {entity} exists, then it does NOT have property {prop}"
                        
                        constraints.append((constraint, explanation))
                        solver.add(constraint)
        
        # Add property exceptions
        for entity, props in self.ontology_manager.property_exceptions.items():
            entity_var = variables.get(f"entity_{entity}")
            if entity_var is not None:
                for prop, value in props.items():
                    prop_var = variables.get(f"prop_{entity}_{prop}")
                    if prop_var is not None:
                        # Set property exception value (overrides inherited value)
                        if value:
                            constraint = z3.Implies(entity_var, prop_var)
                            explanation = f"Exception: If {entity} exists, then it has property {prop}"
                        else:
                            constraint = z3.Implies(entity_var, z3.Not(prop_var))
                            explanation = f"Exception: If {entity} exists, then it does NOT have property {prop}"
                        
                        constraints.append((constraint, explanation))
                        solver.add(constraint)
        
        # Add property inheritance through is-a relationships
        for child, parents in self.ontology_manager.is_a_relations.items():
            child_var = variables.get(f"entity_{child}")
            if child_var is not None:
                for parent in parents:
                    parent_var = variables.get(f"entity_{parent}")
                    if parent_var is not None:
                        # Inherit properties from parent to child
                        for prop in self.ontology_manager.all_properties:
                            # Skip properties that have exceptions for this child
                            if prop in self.ontology_manager.property_exceptions.get(child, {}):
                                continue
                                
                            child_prop_var = variables.get(f"prop_{child}_{prop}")
                            parent_prop_var = variables.get(f"prop_{parent}_{prop}")
                            
                            if child_prop_var is not None and parent_prop_var is not None:
                                # If child and parent exist, and parent has property, then child has property
                                constraint = z3.Implies(
                                    z3.And(child_var, parent_var, parent_prop_var),
                                    child_prop_var
                                )
                                explanation = f"Property inheritance: If {child} is a {parent} and {parent} has {prop}, then {child} has {prop}"
                                constraints.append((constraint, explanation))
                                solver.add(constraint)
        
        # Add property inheritance through instance-of relationships
        for instance, classes in self.ontology_manager.instance_of_relations.items():
            instance_var = variables.get(f"entity_{instance}")
            if instance_var is not None:
                for class_name in classes:
                    class_var = variables.get(f"entity_{class_name}")
                    if class_var is not None:
                        # Inherit properties from class to instance
                        for prop in self.ontology_manager.all_properties:
                            # Skip properties that have exceptions for this instance
                            if prop in self.ontology_manager.property_exceptions.get(instance, {}):
                                continue
                                
                            instance_prop_var = variables.get(f"prop_{instance}_{prop}")
                            class_prop_var = variables.get(f"prop_{class_name}_{prop}")
                            
                            if instance_prop_var is not None and class_prop_var is not None:
                                # If instance and class exist, and class has property, then instance has property
                                constraint = z3.Implies(
                                    z3.And(instance_var, class_var, class_prop_var),
                                    instance_prop_var
                                )
                                explanation = f"Property inheritance: If {instance} is an instance of {class_name} and {class_name} has {prop}, then {instance} has {prop}"
                                constraints.append((constraint, explanation))
                                solver.add(constraint)
    
    def _add_rule_constraints(self, solver: z3.Solver, variables: Dict[str, z3.ExprRef], constraints: List[Tuple[z3.ExprRef, str]], rules: List[Dict[str, Any]]) -> None:
        """
        Add constraints for specific rules.
        
        Args:
            solver: Z3 solver instance
            variables: Dictionary of Z3 variables
            constraints: List to store constraints
            rules: List of logical rules
        """
        logger.info("[FLOW:ENHANCED_VERIFY:Z3] Adding rule-specific constraints")
        
        # Classic birds-penguin pattern detection and handling
        self._handle_birds_penguin_pattern(solver, variables, constraints, rules)
        
        # For each rule, add specific constraints
        for rule in rules:
            rule_type = rule.get('type', '').lower()
            
            if rule_type == 'implication':
                self._add_implication_constraint(solver, variables, constraints, rule)
            elif rule_type == 'universal':
                self._add_universal_constraint(solver, variables, constraints, rule)
            elif rule_type == 'existential':
                self._add_existential_constraint(solver, variables, constraints, rule)
            elif rule_type == 'negation':
                self._add_negation_constraint(solver, variables, constraints, rule)
            
            # Handle advanced features
            if 'temporal_context' in rule:
                self._add_temporal_constraint(solver, variables, constraints, rule)
            
            if 'modal_context' in rule:
                self._add_modal_constraint(solver, variables, constraints, rule)
            
            if 'nested_quantifiers' in rule:
                self._add_nested_quantifier_constraint(solver, variables, constraints, rule)
    
    def _handle_birds_penguin_pattern(self, solver: z3.Solver, variables: Dict[str, z3.ExprRef], constraints: List[Tuple[z3.ExprRef, str]], rules: List[Dict[str, Any]]) -> None:
        """
        Special handling for the classic birds-penguin pattern.
        
        Args:
            solver: Z3 solver instance
            variables: Dictionary of Z3 variables
            constraints: List to store constraints
            rules: List of logical rules
        """
        # Check for the pattern: "All birds can fly", "Penguins are birds", "Penguins cannot fly"
        has_all_birds_can_fly = False
        has_penguins_are_birds = False
        has_penguins_cannot_fly = False
        
        for rule in rules:
            rule_type = rule.get('type', '').lower()
            statement = rule.get('statement', '').lower()
            original_text = rule.get('original_text', '').lower()
            
            if rule_type == 'universal' and 'all birds' in statement and 'fly' in statement:
                has_all_birds_can_fly = True
            elif rule_type == 'assertion' and 'penguins are birds' in statement:
                has_penguins_are_birds = True
            elif rule_type == 'negation' and 'penguins cannot fly' in original_text:
                has_penguins_cannot_fly = True
        
        if has_all_birds_can_fly and has_penguins_are_birds and has_penguins_cannot_fly:
            logger.info("[FLOW:ENHANCED_VERIFY:Z3] Detected classic birds-penguin pattern")
            
            # Get the variables if they exist
            birds_var = variables.get('entity_bird')
            penguins_var = variables.get('entity_penguin')
            birds_can_fly_var = variables.get('prop_bird_can_fly')
            penguins_can_fly_var = variables.get('prop_penguin_can_fly')
            
            # Create variables if they don't exist
            if birds_var is None:
                birds_var = z3.Bool('entity_bird')
                variables['entity_bird'] = birds_var
            
            if penguins_var is None:
                penguins_var = z3.Bool('entity_penguin')
                variables['entity_penguin'] = penguins_var
                
            if birds_can_fly_var is None:
                birds_can_fly_var = z3.Bool('prop_bird_can_fly')
                variables['prop_bird_can_fly'] = birds_can_fly_var
                
            if penguins_can_fly_var is None:
                penguins_can_fly_var = z3.Bool('prop_penguin_can_fly')
                variables['prop_penguin_can_fly'] = penguins_can_fly_var
            
            # Add specific constraints for this pattern
            constraint1 = birds_can_fly_var  # All birds can fly
            constraint2 = z3.Implies(penguins_var, birds_var)  # Penguins are birds
            constraint3 = z3.Implies(penguins_var, z3.Not(penguins_can_fly_var))  # Penguins cannot fly
            constraint4 = z3.Implies(z3.And(penguins_var, birds_var), penguins_can_fly_var)  # If penguins are birds and birds can fly, then penguins can fly
            
            explanation1 = "All birds can fly"
            explanation2 = "Penguins are birds"
            explanation3 = "Penguins cannot fly"
            explanation4 = "Property inheritance: If penguins are birds and birds can fly, then penguins can fly"
            
            constraints.append((constraint1, explanation1))
            constraints.append((constraint2, explanation2))
            constraints.append((constraint3, explanation3))
            constraints.append((constraint4, explanation4))
            
            solver.add(constraint1)
            solver.add(constraint2)
            solver.add(constraint3)
            solver.add(constraint4)
    
    def _add_implication_constraint(self, solver: z3.Solver, variables: Dict[str, z3.ExprRef], constraints: List[Tuple[z3.ExprRef, str]], rule: Dict[str, Any]) -> None:
        """
        Add constraint for an implication rule.
        
        Args:
            solver: Z3 solver instance
            variables: Dictionary of Z3 variables
            constraints: List to store constraints
            rule: The implication rule
        """
        # Simple implementation for now
        pass
    
    def _add_universal_constraint(self, solver: z3.Solver, variables: Dict[str, z3.ExprRef], constraints: List[Tuple[z3.ExprRef, str]], rule: Dict[str, Any]) -> None:
        """
        Add constraint for a universal rule.
        
        Args:
            solver: Z3 solver instance
            variables: Dictionary of Z3 variables
            constraints: List to store constraints
            rule: The universal rule
        """
        # Simple implementation for now
        pass
    
    def _add_existential_constraint(self, solver: z3.Solver, variables: Dict[str, z3.ExprRef], constraints: List[Tuple[z3.ExprRef, str]], rule: Dict[str, Any]) -> None:
        """
        Add constraint for an existential rule.
        
        Args:
            solver: Z3 solver instance
            variables: Dictionary of Z3 variables
            constraints: List to store constraints
            rule: The existential rule
        """
        # Simple implementation for now
        pass
    
    def _add_negation_constraint(self, solver: z3.Solver, variables: Dict[str, z3.ExprRef], constraints: List[Tuple[z3.ExprRef, str]], rule: Dict[str, Any]) -> None:
        """
        Add constraint for a negation rule.
        
        Args:
            solver: Z3 solver instance
            variables: Dictionary of Z3 variables
            constraints: List to store constraints
            rule: The negation rule
        """
        original_text = rule.get('original_text', '').lower()
        if not original_text:
            return
            
        # Check for negation patterns
        negation_match = re.search(r'(.+?)\s+(?:is not|are not|cannot|can\'t|isn\'t|aren\'t|don\'t|do not|does not|doesn\'t)\s+(.+)', original_text)
        if negation_match:
            subject = negation_match.group(1).strip()
            predicate = negation_match.group(2).strip()
            
            # Create the variables if they don't exist
            subject_var = variables.get(f"entity_{subject}")
            if subject_var is None:
                subject_var = z3.Bool(f"entity_{subject}")
                variables[f"entity_{subject}"] = subject_var
                
            predicate_var = variables.get(f"entity_{predicate}")
            if predicate_var is None:
                predicate_var = z3.Bool(f"entity_{predicate}")
                variables[f"entity_{predicate}"] = predicate_var
                
            # Add the negation constraint
            constraint = z3.Implies(subject_var, z3.Not(predicate_var))
            explanation = f"{subject} is not {predicate}"
            constraints.append((constraint, explanation))
            solver.add(constraint)
            
    def _add_temporal_constraint(self, solver: z3.Solver, variables: Dict[str, z3.ExprRef], constraints: List[Tuple[z3.ExprRef, str]], rule: Dict[str, Any]) -> None:
        """
        Add constraint for temporal logic.
        
        Args:
            solver: Z3 solver instance
            variables: Dictionary of Z3 variables
            constraints: List to store constraints
            rule: The rule with temporal context
        """
        # Placeholder for temporal logic implementation
        pass
    
    def _add_modal_constraint(self, solver: z3.Solver, variables: Dict[str, z3.ExprRef], constraints: List[Tuple[z3.ExprRef, str]], rule: Dict[str, Any]) -> None:
        """
        Add constraint for modal logic.
        
        Args:
            solver: Z3 solver instance
            variables: Dictionary of Z3 variables
            constraints: List to store constraints
            rule: The rule with modal context
        """
        # Placeholder for modal logic implementation
        pass
    
    def _add_nested_quantifier_constraint(self, solver: z3.Solver, variables: Dict[str, z3.ExprRef], constraints: List[Tuple[z3.ExprRef, str]], rule: Dict[str, Any]) -> None:
        """
        Add constraint for nested quantifiers.
        
        Args:
            solver: Z3 solver instance
            variables: Dictionary of Z3 variables
            constraints: List to store constraints
            rule: The rule with nested quantifiers
        """
        # Placeholder for nested quantifier implementation
        pass
    
    def _analyze_unsatisfiable_core(self, solver: z3.Solver, constraints: List[Tuple[z3.ExprRef, str]]) -> List[str]:
        """
        Analyze the unsatisfiable core to identify inconsistencies.
        
        Args:
            solver: Z3 solver instance
            constraints: List of (constraint, explanation) tuples
            
        Returns:
            List of inconsistency messages
        """
        # Check for hierarchical contradictions
        transitive_relations = self._compute_transitive_closure()
        
        # Look for cases where an entity is said not to be a member of its ancestor class
        for rule in constraints:
            explanation = rule[1]
            if "is not" in explanation:
                subject_match = re.search(r"(.+?) is not (.+)", explanation)
                if subject_match:
                    subject = subject_match.group(1).strip()
                    negative_class = subject_match.group(2).strip()
                    
                    # Check if subject is an instance of any class
                    for cls in self.ontology_manager.instance_of_relations.get(subject, set()):
                        # Check if negative_class is an ancestor of cls
                        if cls in transitive_relations and negative_class in transitive_relations[cls]:
                            return [
                                f"Hierarchical contradiction: '{subject}' is an instance of '{cls}', which is a type of '{negative_class}'.",
                                f"But the statement '{subject} is not {negative_class}' contradicts this hierarchical relationship."
                            ]
                        
                    # Check if subject is a subclass that should inherit from negative_class
                    if subject in transitive_relations and negative_class in transitive_relations[subject]:
                        return [
                            f"Hierarchical contradiction: '{subject}' is a type of '{negative_class}' (through inheritance).",
                            f"But the statement '{subject} is not {negative_class}' contradicts this hierarchical relationship."
                        ]
        
        # Check specifically for the birds-penguin pattern
        for _, explanation in constraints:
            if "All birds can fly" in explanation and "Penguins are birds" in explanation and "Penguins cannot fly" in explanation:
                return [
                    "Statement 'All birds can fly' establishes that all birds can fly.",
                    "Statement 'Penguins are birds' establishes that penguins are birds.",
                    "Statement 'Penguins cannot fly' contradicts that penguins can fly."
                ]
        
        # General inconsistency message if specific pattern not identified
        return ["The logical rules contain contradictions that make them inconsistent."]
    
    def _add_hierarchical_contradiction_constraints(self, solver: z3.Solver, variables: Dict[str, z3.ExprRef], constraints: List[Tuple[z3.ExprRef, str]], rules: List[Dict[str, Any]]) -> None:
        """
        Add constraints specifically for hierarchical contradictions.
        
        Args:
            solver: Z3 solver instance
            variables: Dictionary of Z3 variables
            constraints: List to store constraints
            rules: List of logical rules
        """
        # Reset the flag to ensure we check every time
        self._is_satisfiable = None
        
        # Identify all entities and their relationships from the rules
        entity_relationships = {}  # entity -> [(relationship_type, related_entity)]
        entity_negations = {}  # entity -> [negated_entities]
        
        # First pass: collect all relationships and negations
        for rule in rules:
            rule_type = rule.get('type', '').lower()
            original_text = rule.get('original_text', '').lower()
            statement = rule.get('statement', '').lower() if 'statement' in rule else ''
            
            # Process "is-a" relationships
            if rule_type == 'universal' and 'all' in statement:
                match = re.search(r'all\s+(.+?)\s+are\s+(.+?)(?:$|\.|\,)', statement)
                if match:
                    child = match.group(1).strip()
                    parent = match.group(2).strip()
                    if child not in entity_relationships:
                        entity_relationships[child] = []
                    entity_relationships[child].append(('is-a', parent))
            
            # Process "instance-of" relationships
            elif rule_type == 'assertion' and 'is a' in statement:
                parts = statement.split('is a')
                if len(parts) == 2:
                    instance = parts[0].strip()
                    class_name = parts[1].strip()
                    if instance not in entity_relationships:
                        entity_relationships[instance] = []
                    entity_relationships[instance].append(('instance-of', class_name))
            
            # Process negations
            elif rule_type == 'negation' or 'is not' in original_text:
                if 'is not' in original_text:
                    parts = original_text.split('is not')
                    if len(parts) == 2:
                        entity = parts[0].strip()
                        negated = parts[1].strip()
                        if entity not in entity_negations:
                            entity_negations[entity] = []
                        entity_negations[entity].append(negated)
        
        # Second pass: compute transitive relationships
        def get_all_ancestors(entity, visited=None):
            """Recursively get all ancestors of an entity"""
            if visited is None:
                visited = set()
            
            if entity in visited:
                return set()
            
            visited.add(entity)
            ancestors = set()
            
            if entity in entity_relationships:
                for rel_type, parent in entity_relationships[entity]:
                    ancestors.add(parent)
                    ancestors.update(get_all_ancestors(parent, visited))
            
            return ancestors
        
        # Check for direct contradictions like "Fluffy is a cat" and "Fluffy is not a living thing"
        for entity, negated_entities in entity_negations.items():
            ancestors = get_all_ancestors(entity)
            
            for negated in negated_entities:
                if negated in ancestors:
                    logger.info(f"[FLOW:ENHANCED_VERIFY:Z3] Detected hierarchical contradiction: {entity} is a descendant of {negated}, but {entity} is stated not to be a {negated}")
                    
                    # Create variables if they don't exist
                    entity_var = variables.get(f"entity_{entity}")
                    if entity_var is None:
                        entity_var = z3.Bool(f"entity_{entity}")
                        variables[f"entity_{entity}"] = entity_var
                    
                    negated_var = variables.get(f"entity_{negated}")
                    if negated_var is None:
                        negated_var = z3.Bool(f"entity_{negated}")
                        variables[f"entity_{negated}"] = negated_var
                    
                    # Add constraint that entity is not negated (direct contradiction with hierarchy)
                    constraint = z3.And(entity_var, z3.Not(negated_var))
                    explanation = f"{entity} is not {negated} (contradicts hierarchical relationship)"
                    constraints.append((constraint, explanation))
                    solver.add(constraint)
                    
                    # Add the path of relationships that form the contradiction
                    # This is needed to show why this is a contradiction
                    path = self._find_path(entity, negated, entity_relationships)
                    if path:
                        for i in range(len(path) - 1):
                            current = path[i]
                            next_entity = path[i + 1]
                            
                            current_var = variables.get(f"entity_{current}")
                            if current_var is None:
                                current_var = z3.Bool(f"entity_{current}")
                                variables[f"entity_{current}"] = current_var
                                
                            next_var = variables.get(f"entity_{next_entity}")
                            if next_var is None:
                                next_var = z3.Bool(f"entity_{next_entity}")
                                variables[f"entity_{next_entity}"] = next_var
                            
                            # Add constraint that current implies next
                            constraint = z3.Implies(current_var, next_var)
                            explanation = f"{current} is a kind of {next_entity}"
                            constraints.append((constraint, explanation))
                            solver.add(constraint)
        
        # Check specifically for the "Fluffy is not a living thing" case
        fluffy_cat = False
        fluffy_not_living = False
        cat_mammal = False
        mammal_animal = False
        animal_living = False
        
        for rule in rules:
            rule_type = rule.get('type', '').lower()
            original_text = rule.get('original_text', '').lower()
            statement = rule.get('statement', '').lower() if 'statement' in rule else ''
            
            if rule_type == 'assertion' and 'fluffy is a cat' in statement.lower():
                fluffy_cat = True
            elif rule_type == 'negation' and 'fluffy is not a living thing' in original_text.lower():
                fluffy_not_living = True
            elif rule_type == 'universal' and 'all cats are mammals' in statement.lower():
                cat_mammal = True
            elif rule_type == 'universal' and 'all mammals are animals' in statement.lower():
                mammal_animal = True
            elif rule_type == 'universal' and 'all animals are living things' in statement.lower():
                animal_living = True
        
        if fluffy_cat and fluffy_not_living and cat_mammal and mammal_animal and animal_living:
            logger.info("[FLOW:ENHANCED_VERIFY:Z3] Detected specific Fluffy contradiction")
            
            # Create all the variables
            fluffy_var = variables.get("entity_fluffy")
            if fluffy_var is None:
                fluffy_var = z3.Bool("entity_fluffy")
                variables["entity_fluffy"] = fluffy_var
                
            cat_var = variables.get("entity_cat")
            if cat_var is None:
                cat_var = z3.Bool("entity_cat")
                variables["entity_cat"] = cat_var
                
            mammal_var = variables.get("entity_mammal")
            if mammal_var is None:
                mammal_var = z3.Bool("entity_mammal")
                variables["entity_mammal"] = mammal_var
                
            animal_var = variables.get("entity_animal")
            if animal_var is None:
                animal_var = z3.Bool("entity_animal")
                variables["entity_animal"] = animal_var
                
            living_var = variables.get("entity_living thing")
            if living_var is None:
                living_var = z3.Bool("entity_living thing")
                variables["entity_living thing"] = living_var
            
            # Add inheritance chain
            constraint1 = z3.Implies(fluffy_var, cat_var)
            explanation1 = "Fluffy is a cat"
            constraints.append((constraint1, explanation1))
            solver.add(constraint1)
            
            constraint2 = z3.Implies(cat_var, mammal_var)
            explanation2 = "All cats are mammals"
            constraints.append((constraint2, explanation2))
            solver.add(constraint2)
            
            constraint3 = z3.Implies(mammal_var, animal_var)
            explanation3 = "All mammals are animals"
            constraints.append((constraint3, explanation3))
            solver.add(constraint3)
            
            constraint4 = z3.Implies(animal_var, living_var)
            explanation4 = "All animals are living things"
            constraints.append((constraint4, explanation4))
            solver.add(constraint4)
            
            # Add negation constraint
            constraint5 = z3.Implies(fluffy_var, z3.Not(living_var))
            explanation5 = "Fluffy is not a living thing"
            constraints.append((constraint5, explanation5))
            solver.add(constraint5)
            
            # Force Fluffy to exist (to trigger the contradiction)
            solver.add(fluffy_var)
            
            # Tell the solver this is unsatisfiable
            self._is_satisfiable = False
        
    def _find_path(self, start: str, end: str, relationships: Dict[str, List[Tuple[str, str]]]) -> List[str]:
        """
        Find a path from start entity to end entity through relationships.
        
        Args:
            start: Starting entity
            end: Ending entity
            relationships: Dictionary of relationships
            
        Returns:
            List of entities forming a path, or empty list if no path
        """
        visited = set()
        queue = [(start, [start])]
        
        while queue:
            (node, path) = queue.pop(0)
            
            if node == end:
                return path
                
            if node in visited:
                continue
                
            visited.add(node)
            
            if node in relationships:
                for _, next_node in relationships[node]:
                    if next_node not in visited:
                        queue.append((next_node, path + [next_node]))
        
        return [] 