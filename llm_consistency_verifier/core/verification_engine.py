import logging
import time
from typing import Dict, List, Optional, Tuple, Set, Any, Union
from enum import Enum
import z3
import sympy
from ..config.config import Config
from ..models.logic_model import LogicalRule, Formula, LogicalOperator

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

class VerificationResult:
    """Result of a verification process."""
    
    def __init__(self, is_consistent: bool, inconsistencies: List[str] = None, proof: str = None):
        self.is_consistent = is_consistent
        self.inconsistencies = inconsistencies or []
        self.proof = proof
        self.verification_time = None
    
    def __str__(self) -> str:
        if self.is_consistent:
            return f"Verification Result: Consistent (time: {self.verification_time:.2f}s)"
        else:
            inconsistencies_str = "\n  ".join(self.inconsistencies)
            return f"Verification Result: Inconsistent (time: {self.verification_time:.2f}s)\nInconsistencies:\n  {inconsistencies_str}"

class SolverType(str, Enum):
    """Types of solvers supported."""
    Z3 = "z3"
    SYMPY = "sympy"

class VerificationEngine:
    """Engine for verifying logical consistency."""
    
    def __init__(self, solver_type: str = Config.SOLVER_TYPE):
        """Initialize the verification engine."""
        self.solver_type = SolverType(solver_type)
        logger.info(f"Verification Engine initialized with solver: {solver_type}")
    
    def verify(self, rules: List[Dict[str, Any]]) -> VerificationResult:
        """Verify the consistency of a set of logical rules."""
        start_time = time.time()
        
        logger.info(f"Starting verification of {len(rules)} rules")
        
        # Convert parsed rules to formal logical representations
        formalized_rules = self._formalize_rules(rules)
        
        # Check for consistency using the appropriate solver
        if self.solver_type == SolverType.Z3:
            result = self._verify_with_z3(formalized_rules, rules)
        elif self.solver_type == SolverType.SYMPY:
            result = self._verify_with_sympy(formalized_rules, rules)
        else:
            raise ValueError(f"Unsupported solver: {self.solver_type}")
        
        # Record verification time
        result.verification_time = time.time() - start_time
        logger.info(f"Verification completed in {result.verification_time:.2f}s: {'Consistent' if result.is_consistent else 'Inconsistent'}")
        
        return result
    
    def _formalize_rules(self, rules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Convert parsed rules to formal representations for the solver."""
        # This is a simplified implementation
        # A full implementation would convert to proper formal logic
        
        formalized = {}
        
        # Create symbols for each unique entity
        entities = set()
        for rule in rules:
            if rule["type"] == "implication":
                entities.add(rule["antecedent"])
                entities.add(rule["consequent"])
            elif rule["type"] in ["universal", "negation", "assertion"]:
                entities.add(rule["statement"])
        
        formalized["entities"] = list(entities)
        formalized["rules"] = rules
        
        return formalized
    
    def _verify_with_z3(self, formalized_rules: Dict[str, Any], original_rules: List[Dict[str, Any]]) -> VerificationResult:
        """Verify logical consistency using Z3 solver."""
        solver = z3.Solver()
        z3_vars = {}
        inconsistencies = []
        
        # Create Z3 boolean variables for each entity
        for entity in formalized_rules["entities"]:
            z3_vars[entity] = z3.Bool(entity.replace(" ", "_"))
        
        # Add constraints based on rules
        for rule in original_rules:
            if rule["type"] == "implication":
                antecedent = rule["antecedent"]
                consequent = rule["consequent"]
                
                # Create implication constraint: antecedent => consequent
                if antecedent in z3_vars and consequent in z3_vars:
                    solver.add(z3.Implies(z3_vars[antecedent], z3_vars[consequent]))
                
            elif rule["type"] == "universal":
                # Handle universal quantification
                statement = rule["statement"]
                if statement in z3_vars:
                    solver.add(z3_vars[statement])
                    
            elif rule["type"] == "negation":
                # Handle negation properly
                statement = rule["statement"]
                # Extract the actual statement being negated
                negated_part = statement.lower()
                if "not" in negated_part:
                    negated_part = negated_part.replace("not", "").strip()
                    if negated_part in z3_vars:
                        solver.add(z3.Not(z3_vars[negated_part]))
                    
            elif rule["type"] == "assertion":
                statement = rule["statement"]
                if statement in z3_vars:
                    solver.add(z3_vars[statement])
        
        # Check satisfiability
        result = solver.check()
        
        if result == z3.sat:
            # Get the model to verify there are no hidden contradictions
            model = solver.model()
            logger.debug("Z3 solver found a satisfying assignment")
            return VerificationResult(True)
        else:
            logger.debug("Z3 solver found an inconsistency")
            
            # Extract core inconsistencies
            unsat_core = solver.unsat_core()
            
            # Identify conflicting rules
            for i, rule1 in enumerate(original_rules):
                for rule2 in original_rules[i+1:]:
                    # Check for direct contradictions
                    if rule1["type"] == "assertion" and rule2["type"] == "negation":
                        if rule1["statement"] in rule2["statement"]:
                            inconsistencies.append(f"Contradiction between '{rule1['original_text']}' and '{rule2['original_text']}'")
                    
                    elif rule1["type"] == "negation" and rule2["type"] == "assertion":
                        if rule2["statement"] in rule1["statement"]:
                            inconsistencies.append(f"Contradiction between '{rule1['original_text']}' and '{rule2['original_text']}'")
                    
                    # Check for conflicting implications
                    elif rule1["type"] == "implication" and rule2["type"] == "implication":
                        if (rule1["antecedent"] == rule2["antecedent"] and 
                            rule1["consequent"] != rule2["consequent"]):
                            inconsistencies.append(f"Conflicting implications: '{rule1['original_text']}' and '{rule2['original_text']}'")
                        
                        # Check for cyclic implications that lead to contradictions
                        elif (rule1["consequent"] == rule2["antecedent"] and
                              rule2["consequent"] == rule1["antecedent"]):
                            inconsistencies.append(f"Cyclic implications that may lead to contradiction: '{rule1['original_text']}' and '{rule2['original_text']}'")
            
            # If no specific inconsistencies found, provide a general message
            if not inconsistencies:
                inconsistencies.append("The set of statements is logically inconsistent")
            
            return VerificationResult(False, inconsistencies)
    
    def _verify_with_sympy(self, formalized_rules: Dict[str, Any], original_rules: List[Dict[str, Any]]) -> VerificationResult:
        """Verify logical consistency using SymPy."""
        # This is a placeholder implementation
        # A full implementation would use SymPy's logical inference capabilities
        logger.warning("SymPy verification not fully implemented, falling back to Z3")
        return self._verify_with_z3(formalized_rules, original_rules) 