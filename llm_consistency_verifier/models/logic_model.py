from enum import Enum
from typing import List, Dict, Optional, Union, Set
from pydantic import BaseModel, Field
import re

class LogicalOperator(str, Enum):
    """Enumeration of logical operators."""
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    IMPLIES = "IMPLIES"
    EQUIVALENT = "EQUIVALENT"
    FORALL = "FORALL"
    EXISTS = "EXISTS"


class Term(BaseModel):
    """Model for representing a logical term (constant, variable, or function)."""
    name: str
    arguments: List["Term"] = Field(default_factory=list)
    is_variable: bool = False
    is_function: bool = False
    
    def __str__(self) -> str:
        if not self.arguments:
            return self.name
        args_str = ", ".join(str(arg) for arg in self.arguments)
        return f"{self.name}({args_str})"


class Predicate(BaseModel):
    """Model for representing a logical predicate."""
    name: str
    arguments: List[Term] = Field(default_factory=list)
    
    def __str__(self) -> str:
        if not self.arguments:
            return self.name
        args_str = ", ".join(str(arg) for arg in self.arguments)
        return f"{self.name}({args_str})"


class Formula(BaseModel):
    """Model for representing a logical formula."""
    operator: Optional[LogicalOperator] = None
    predicate: Optional[Predicate] = None
    subformulas: List["Formula"] = Field(default_factory=list)
    quantifier_variable: Optional[Term] = None
    
    def __str__(self) -> str:
        if self.predicate:
            return str(self.predicate)
        
        if self.operator == LogicalOperator.NOT:
            return f"¬({str(self.subformulas[0])})"
        
        if self.operator in [LogicalOperator.FORALL, LogicalOperator.EXISTS]:
            quantifier = "∀" if self.operator == LogicalOperator.FORALL else "∃"
            return f"{quantifier}{str(self.quantifier_variable)}({str(self.subformulas[0])})"
        
        op_symbols = {
            LogicalOperator.AND: "∧",
            LogicalOperator.OR: "∨",
            LogicalOperator.IMPLIES: "→",
            LogicalOperator.EQUIVALENT: "↔"
        }
        
        subformula_strs = [str(f) for f in self.subformulas]
        return f"({op_symbols[self.operator].join(subformula_strs)})"


class LogicalRule(BaseModel):
    """Model for representing a logical rule extracted from text."""
    formula: Formula
    original_text: str
    source_position: Optional[int] = None
    confidence: float = 1.0


class LogicKnowledgeBase:
    """Knowledge base containing logical formulas and rules."""
    
    def __init__(self):
        self.rules: List[LogicalRule] = []
        self.formulas: List[Formula] = []
        self.terms: Dict[str, Term] = {}
        self.predicates: Dict[str, Set[Predicate]] = {}
    
    def add_rule(self, rule: LogicalRule) -> None:
        """Add a rule to the knowledge base."""
        self.rules.append(rule)
        self.formulas.append(rule.formula)
        self._extract_predicates_and_terms(rule.formula)
    
    def _extract_predicates_and_terms(self, formula: Formula) -> None:
        """Extract and store predicates and terms from a formula."""
        if formula.predicate:
            pred_name = formula.predicate.name
            if pred_name not in self.predicates:
                self.predicates[pred_name] = set()
            self.predicates[pred_name].add(formula.predicate)
            
            for term in formula.predicate.arguments:
                self._store_term(term)
        
        if formula.quantifier_variable:
            self._store_term(formula.quantifier_variable)
            
        for subformula in formula.subformulas:
            self._extract_predicates_and_terms(subformula)
    
    def _store_term(self, term: Term) -> None:
        """Store a term in the knowledge base."""
        if term.name not in self.terms:
            self.terms[term.name] = term
            
        for arg in term.arguments:
            self._store_term(arg)
    
    def get_formulas(self) -> List[Formula]:
        """Get all formulas in the knowledge base."""
        return self.formulas.copy()
    
    def get_rules(self) -> List[LogicalRule]:
        """Get all rules in the knowledge base."""
        return self.rules.copy()
    
    def get_predicates(self) -> Dict[str, Set[Predicate]]:
        """Get all predicates grouped by name."""
        return self.predicates.copy()
    
    def get_terms(self) -> Dict[str, Term]:
        """Get all terms indexed by name."""
        return self.terms.copy()


class NaturalLanguageParser:
    """Parser for converting natural language into logical formulas."""
    
    @staticmethod
    def extract_rules(text: str) -> List[LogicalRule]:
        """
        Extract logical rules from natural language text.
        This is a simplified placeholder implementation.
        A full implementation would use NLP techniques or an LLM.
        """
        rules = []
        
        # Simplified pattern matching for demonstration
        # In a real implementation, this would be replaced with more sophisticated NLP
        
        # Match "All X are Y" pattern
        all_pattern = r"All ([a-zA-Z\s]+) (?:are|is) ([a-zA-Z\s]+)"
        for match in re.finditer(all_pattern, text):
            x, y = match.groups()
            x, y = x.strip(), y.strip()
            
            # Create terms
            x_term = Term(name=x.replace(" ", "_"))
            
            # Create predicate X(v) -> Y(v)
            x_pred = Predicate(name=x.replace(" ", "_"), arguments=[Term(name="v", is_variable=True)])
            y_pred = Predicate(name=y.replace(" ", "_"), arguments=[Term(name="v", is_variable=True)])
            
            # Create formula: ∀v(X(v) → Y(v))
            implies_formula = Formula(
                operator=LogicalOperator.IMPLIES,
                subformulas=[
                    Formula(predicate=x_pred),
                    Formula(predicate=y_pred)
                ]
            )
            
            forall_formula = Formula(
                operator=LogicalOperator.FORALL,
                quantifier_variable=Term(name="v", is_variable=True),
                subformulas=[implies_formula]
            )
            
            rule = LogicalRule(
                formula=forall_formula,
                original_text=match.group(0),
                source_position=match.start()
            )
            
            rules.append(rule)
        
        # Match "X is Y" pattern (for specific instances)
        is_pattern = r"([a-zA-Z\s]+) (?:is a|is an|is|are) ([a-zA-Z\s]+)"
        for match in re.finditer(is_pattern, text):
            x, y = match.groups()
            x, y = x.strip(), y.strip()
            
            # Skip if it matches the "All X are Y" pattern
            if re.match(r"All ", x):
                continue
                
            # Create predicates X(x) and Y(x)
            x_term = Term(name=x.replace(" ", "_"))
            y_pred = Predicate(name=y.replace(" ", "_"), arguments=[x_term])
            
            formula = Formula(predicate=y_pred)
            
            rule = LogicalRule(
                formula=formula,
                original_text=match.group(0),
                source_position=match.start()
            )
            
            rules.append(rule)
        
        # Match "X is not Y" or "X cannot Y" pattern
        not_pattern = r"([a-zA-Z\s]+) (?:is not|cannot|can't|doesn't|does not) ([a-zA-Z\s]+)"
        for match in re.finditer(not_pattern, text):
            x, y = match.groups()
            x, y = x.strip(), y.strip()
            
            # Create predicates X(x) and Y(x)
            x_term = Term(name=x.replace(" ", "_"))
            y_pred = Predicate(name=y.replace(" ", "_"), arguments=[x_term])
            
            formula = Formula(
                operator=LogicalOperator.NOT,
                subformulas=[Formula(predicate=y_pred)]
            )
            
            rule = LogicalRule(
                formula=formula,
                original_text=match.group(0),
                source_position=match.start()
            )
            
            rules.append(rule)
            
        return rules
