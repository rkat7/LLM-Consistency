import re
import logging
from typing import List, Dict, Any, Set, Tuple, Optional
from ..config.config import Config
from ..models.logic_model import Term, Predicate, Formula, LogicalRule, LogicalOperator

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

class RuleExtractor:
    """
    Extracts logical rules from text using NLP techniques without LLM dependency.
    """
    
    # Common words to ignore when identifying entities
    STOPWORDS = {'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'in', 'on', 'at', 
                'to', 'for', 'with', 'by', 'from', 'up', 'down', 'out', 'as', 'is',
                'are', 'be', 'was', 'were', 'been', 'being', 'have', 'has', 'had',
                'do', 'does', 'did', 'can', 'could', 'will', 'would', 'shall',
                'should', 'may', 'might', 'must', 'that', 'which', 'who', 'whom',
                'whose', 'this', 'these', 'those', 'of', 'while', 'when'}
    
    # Patterns for identifying different types of logical statements
    UNIVERSAL_PATTERNS = [
        r'\b(all|every|any|each)\s+([^.!?]+?)\s+(are|is|have|has)\s+([^.!?]+)',
        r'([^.!?]+?)\s+is\s+always\s+([^.!?]+)',
    ]
    
    EXISTENTIAL_PATTERNS = [
        r'\b(some|there\s+(?:are|is|exists))\s+([^.!?]+?)\s+(?:that|who|which)\s+([^.!?]+)',
        r'\b(some|there\s+(?:are|is|exists))\s+([^.!?]+)',
    ]
    
    IMPLICATION_PATTERNS = [
        r'\bif\s+([^,]+?),?\s+then\s+([^.!?]+)',
        r'([^.!?]+?)\s+implies\s+([^.!?]+)',
        r'([^.!?]+?)\s+(?:→|->)\s+([^.!?]+)',
        r'when(ever)?\s+([^,]+?),\s+([^.!?]+)',
    ]
    
    NEGATION_PATTERNS = [
        r'([^.!?]+?)\s+(?:is not|are not|cannot|can\'t|doesn\'t|does not|don\'t|do not)\s+([^.!?]+)',
        r'(?:no|not all)\s+([^.!?]+?)\s+(?:are|is)\s+([^.!?]+)',
        r'it\s+is\s+not\s+the\s+case\s+that\s+([^.!?]+)',
    ]
    
    ASSERTION_PATTERNS = [
        r'([^.!?]+?)\s+(?:is|are)\s+([^.!?]+)',
        r'([^.!?]+?)\s+(?:has|have)\s+([^.!?]+)',
    ]
    
    def __init__(self):
        """Initialize the rule extractor."""
        logger.info("Initializing rule extractor with NLP-based techniques")
    
    def extract_rules(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract logical rules from text using pattern matching and NLP techniques.
        
        Args:
            text: The text to extract rules from
            
        Returns:
            List of extracted rules in dictionary format
        """
        # Preprocess text
        text = self._preprocess_text(text)
        
        # Extract sentences
        sentences = self._extract_sentences(text)
        logger.debug(f"Extracted {len(sentences)} sentences")
        
        rules = []
        
        # Process each sentence to extract rules
        for sentence in sentences:
            # Extract universal rules (All X are Y)
            universal_rules = self._extract_universal_rules(sentence)
            if universal_rules:
                rules.extend(universal_rules)
                continue
                
            # Extract implications (If X then Y)
            implication_rules = self._extract_implication_rules(sentence)
            if implication_rules:
                rules.extend(implication_rules)
                continue
                
            # Extract negations (X is not Y)
            negation_rules = self._extract_negation_rules(sentence)
            if negation_rules:
                rules.extend(negation_rules)
                continue
                
            # Extract existential rules (Some X are Y)
            existential_rules = self._extract_existential_rules(sentence)
            if existential_rules:
                rules.extend(existential_rules)
                continue
                
            # Extract general assertions (X is Y)
            assertion_rules = self._extract_assertion_rules(sentence)
            if assertion_rules:
                rules.extend(assertion_rules)
        
        # Post-process to normalize entities and resolve coreferences
        rules = self._post_process_rules(rules)
        
        logger.info(f"Extracted {len(rules)} logical rules from text")
        return rules
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for rule extraction."""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Normalize punctuation
        text = text.replace('...', '.')
        text = re.sub(r'\.+', '.', text)
        
        # Ensure proper spacing after punctuation
        text = re.sub(r'(\.)([A-Za-z])', r'\1 \2', text)
        
        # Replace some common abbreviations
        text = text.replace("can't", "cannot")
        text = text.replace("won't", "will not")
        text = text.replace("n't", " not")
        
        return text
    
    def _extract_sentences(self, text: str) -> List[str]:
        """Extract individual sentences from text."""
        # Simple sentence splitting by punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_universal_rules(self, sentence: str) -> List[Dict[str, Any]]:
        """Extract universal rules from a sentence."""
        rules = []
        sentence_lower = sentence.lower()
        
        for pattern in self.UNIVERSAL_PATTERNS:
            matches = re.finditer(pattern, sentence_lower)
            for match in matches:
                try:
                    if match.group(1) in ['all', 'every', 'any', 'each']:
                        subject = match.group(2).strip()
                        predicate = match.group(4).strip()
                    else:
                        subject = match.group(1).strip()
                        predicate = match.group(2).strip()
                    
                    rule = {
                        "type": "universal",
                        "statement": f"All {subject} are {predicate}",
                        "original_text": sentence
                    }
                    rules.append(rule)
                except (IndexError, AttributeError):
                    continue
        
        return rules
    
    def _extract_implication_rules(self, sentence: str) -> List[Dict[str, Any]]:
        """Extract implication rules from a sentence."""
        rules = []
        sentence_lower = sentence.lower()
        
        for pattern in self.IMPLICATION_PATTERNS:
            matches = re.finditer(pattern, sentence_lower)
            for match in matches:
                try:
                    antecedent = match.group(1).strip() if 'when' not in pattern else match.group(2).strip()
                    consequent = match.group(2).strip() if 'when' not in pattern else match.group(3).strip()
                    
                    rule = {
                        "type": "implication",
                        "antecedent": antecedent,
                        "consequent": consequent,
                        "original_text": sentence
                    }
                    rules.append(rule)
                except (IndexError, AttributeError):
                    continue
        
        return rules
    
    def _extract_negation_rules(self, sentence: str) -> List[Dict[str, Any]]:
        """Extract negation rules from a sentence."""
        rules = []
        sentence_lower = sentence.lower()
        
        for pattern in self.NEGATION_PATTERNS:
            matches = re.finditer(pattern, sentence_lower)
            for match in matches:
                try:
                    if 'it is not the case' in pattern:
                        statement = f"NOT ({match.group(1).strip()})"
                    else:
                        subject = match.group(1).strip()
                        predicate = match.group(2).strip() if len(match.groups()) > 1 else ""
                        statement = f"{subject} is not {predicate}" if predicate else subject
                    
                    rule = {
                        "type": "negation",
                        "statement": statement,
                        "original_text": sentence
                    }
                    rules.append(rule)
                except (IndexError, AttributeError):
                    continue
        
        return rules
    
    def _extract_existential_rules(self, sentence: str) -> List[Dict[str, Any]]:
        """Extract existential rules from a sentence."""
        rules = []
        sentence_lower = sentence.lower()
        
        for pattern in self.EXISTENTIAL_PATTERNS:
            matches = re.finditer(pattern, sentence_lower)
            for match in matches:
                try:
                    # Handle different pattern structures
                    if len(match.groups()) >= 3:
                        quantifier = match.group(1).strip()
                        subject = match.group(2).strip()
                        predicate = match.group(3).strip()
                        statement = f"{quantifier} {subject} {predicate}"
                    else:
                        quantifier = match.group(1).strip()
                        subject = match.group(2).strip()
                        statement = f"{quantifier} {subject}"
                    
                    rule = {
                        "type": "existential",
                        "statement": statement,
                        "original_text": sentence
                    }
                    rules.append(rule)
                except (IndexError, AttributeError):
                    continue
        
        return rules
    
    def _extract_assertion_rules(self, sentence: str) -> List[Dict[str, Any]]:
        """Extract general assertion rules from a sentence."""
        rules = []
        sentence_lower = sentence.lower()
        
        for pattern in self.ASSERTION_PATTERNS:
            matches = re.finditer(pattern, sentence_lower)
            for match in matches:
                try:
                    subject = match.group(1).strip()
                    predicate = match.group(2).strip()
                    
                    # Skip if subject is a stopword or too short
                    if subject.lower() in self.STOPWORDS or len(subject) < 2:
                        continue
                        
                    rule = {
                        "type": "assertion",
                        "statement": f"{subject} is {predicate}",
                        "original_text": sentence
                    }
                    rules.append(rule)
                except (IndexError, AttributeError):
                    continue
        
        # If no structured assertions found, use the entire sentence as a general assertion
        if not rules and not self._is_special_form(sentence_lower):
            rules.append({
                "type": "assertion",
                "statement": sentence,
                "original_text": sentence
            })
        
        return rules
    
    def _is_special_form(self, sentence: str) -> bool:
        """Check if sentence matches any special logical form patterns."""
        # Check all defined patterns to see if this is a special form
        for pattern in (self.UNIVERSAL_PATTERNS + self.EXISTENTIAL_PATTERNS + 
                      self.IMPLICATION_PATTERNS + self.NEGATION_PATTERNS):
            if re.search(pattern, sentence):
                return True
        return False
    
    def _post_process_rules(self, rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Post-process extracted rules to normalize and improve quality."""
        processed_rules = []
        rule_texts = set()  # To avoid duplicates
        
        for rule in rules:
            # Skip empty or invalid rules
            if not rule:
                continue
                
            # Normalize rule text representation
            if rule["type"] == "implication":
                rule_text = f"{rule['antecedent']} -> {rule['consequent']}"
            else:
                rule_text = rule.get("statement", "")
            
            # Skip if we've already seen this rule
            if rule_text in rule_texts:
                continue
                
            rule_texts.add(rule_text)
            processed_rules.append(rule)
        
        return processed_rules


class SpacyRuleExtractor(RuleExtractor):
    """
    Enhanced rule extractor using spaCy for better NLP capabilities.
    """
    
    def __init__(self):
        """Initialize the spaCy-based rule extractor."""
        super().__init__()
        self._initialize_spacy()
    
    def _initialize_spacy(self):
        """Initialize spaCy model and components."""
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Initialized spaCy model for rule extraction")
            self.use_spacy = True
        except (ImportError, OSError):
            logger.warning("spaCy or language model not available, falling back to regex patterns")
            self.use_spacy = False
    
    def extract_rules(self, text: str) -> List[Dict[str, Any]]:
        """Extract rules using spaCy for enhanced NLP if available."""
        # If spaCy is not available, fall back to pattern-based extraction
        if not getattr(self, 'use_spacy', False):
            return super().extract_rules(text)
        
        # Preprocess text
        text = self._preprocess_text(text)
        
        # Parse with spaCy
        doc = self.nlp(text)
        
        rules = []
        
        # Process each sentence
        for sent in doc.sents:
            # Extract rules using dependency parsing
            sent_rules = self._extract_rules_from_dependencies(sent)
            if sent_rules:
                rules.extend(sent_rules)
            else:
                # Fall back to pattern matching if dependency parsing didn't yield results
                sent_text = sent.text
                sent_rules = []
                sent_rules.extend(self._extract_universal_rules(sent_text))
                sent_rules.extend(self._extract_implication_rules(sent_text))
                sent_rules.extend(self._extract_negation_rules(sent_text))
                sent_rules.extend(self._extract_existential_rules(sent_text))
                sent_rules.extend(self._extract_assertion_rules(sent_text))
                rules.extend(sent_rules)
        
        # Post-process to normalize and remove duplicates
        rules = self._post_process_rules(rules)
        
        logger.info(f"Extracted {len(rules)} logical rules from text using spaCy")
        return rules
    
    def _extract_rules_from_dependencies(self, sent) -> List[Dict[str, Any]]:
        """Extract logical rules using dependency parsing."""
        rules = []
        
        # Get the root verb of the sentence
        root = None
        for token in sent:
            if token.dep_ == "ROOT":
                root = token
                break
        
        if not root:
            return []
        
        # Extract implication rules from conditional structures
        if any(token.dep_ == "mark" and token.text.lower() == "if" for token in sent):
            return self._extract_implication_from_deps(sent)
        
        # Extract negation rules
        if any(token.dep_ == "neg" for token in sent):
            return self._extract_negation_from_deps(sent, root)
        
        # Extract universal quantification
        if any(token.text.lower() in ("all", "every", "each") for token in sent):
            return self._extract_universal_from_deps(sent)
        
        # Extract simple assertions for anything else
        return self._extract_assertion_from_deps(sent, root)
    
    def _extract_implication_from_deps(self, sent) -> List[Dict[str, Any]]:
        """Extract implication rules from dependency parsed sentence."""
        rules = []
        
        # Find the "if" clause and main clause
        if_clause = ""
        then_clause = ""
        
        # Very simplified approach - a real implementation would be more robust
        for token in sent:
            if token.dep_ == "mark" and token.text.lower() == "if":
                # Find the clause that contains this token
                if_head = token.head
                if_subtree = list(if_head.subtree)
                if_start = min(t.i for t in if_subtree)
                if_end = max(t.i for t in if_subtree)
                if_clause = sent[if_start:if_end+1].text
                
                # The rest is likely the then clause
                then_tokens = [t for t in sent if t.i > if_end]
                if then_tokens:
                    then_clause = " ".join(t.text for t in then_tokens)
        
        if if_clause and then_clause:
            # Clean up clauses
            if_clause = if_clause.lower().replace("if ", "", 1).strip()
            then_clause = then_clause.strip()
            
            rules.append({
                "type": "implication",
                "antecedent": if_clause,
                "consequent": then_clause,
                "original_text": sent.text
            })
        
        return rules
    
    def _extract_negation_from_deps(self, sent, root) -> List[Dict[str, Any]]:
        """Extract negation rules from dependency parsed sentence."""
        rules = []
        
        # Check for negation
        for token in sent:
            if token.dep_ == "neg" and token.head == root:
                # Extract subject and predicate
                subjects = []
                predicates = []
                
                for child in root.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        # Get the full noun phrase
                        subjects.append(" ".join(t.text for t in child.subtree))
                    elif child.dep_ in ("dobj", "attr", "acomp"):
                        # Get the full object or complement
                        predicates.append(" ".join(t.text for t in child.subtree))
                
                if subjects:
                    subject = subjects[0]
                    predicate = predicates[0] if predicates else ""
                    
                    statement = f"{subject} is not {predicate}" if predicate else f"NOT ({subject})"
                    
                    rules.append({
                        "type": "negation",
                        "statement": statement,
                        "original_text": sent.text
                    })
        
        return rules
    
    def _extract_universal_from_deps(self, sent) -> List[Dict[str, Any]]:
        """Extract universal quantification rules from dependency parsed sentence."""
        rules = []
        
        # Find the quantifier
        quantifier = None
        for token in sent:
            if token.text.lower() in ("all", "every", "each"):
                quantifier = token
                break
        
        if quantifier:
            # Get the noun being quantified
            noun = None
            for token in sent:
                if token.head == quantifier and token.dep_ in ("pobj", "dobj"):
                    noun = " ".join(t.text for t in token.subtree)
                    break
            
            # Get the predicate
            root = None
            for token in sent:
                if token.dep_ == "ROOT":
                    root = token
                    break
            
            predicate = ""
            if root and root.pos_ == "VERB":
                predicate_tokens = []
                for token in sent:
                    if token.head == root and token.dep_ in ("dobj", "attr", "acomp"):
                        predicate_tokens.extend(t.text for t in token.subtree)
                
                predicate = " ".join(predicate_tokens)
            
            if noun and predicate:
                rules.append({
                    "type": "universal",
                    "statement": f"All {noun} are {predicate}",
                    "original_text": sent.text
                })
        
        return rules
    
    def _extract_assertion_from_deps(self, sent, root) -> List[Dict[str, Any]]:
        """Extract simple assertion rules from dependency parsed sentence."""
        rules = []
        
        if root and root.pos_ == "VERB":
            # Get subject
            subjects = []
            for token in sent:
                if token.head == root and token.dep_ in ("nsubj", "nsubjpass"):
                    subjects.append(" ".join(t.text for t in token.subtree))
            
            # Get predicate
            predicates = []
            for token in sent:
                if token.head == root and token.dep_ in ("dobj", "attr", "acomp", "xcomp"):
                    predicates.append(" ".join(t.text for t in token.subtree))
            
            if subjects and (predicates or root.text.lower() != "is"):
                subject = subjects[0]
                predicate = predicates[0] if predicates else root.text
                
                rules.append({
                    "type": "assertion",
                    "statement": f"{subject} {root.text} {predicate}",
                    "original_text": sent.text
                })
            elif subjects:
                # For simple statements like "X exists"
                rules.append({
                    "type": "assertion",
                    "statement": f"{subjects[0]} {root.text}",
                    "original_text": sent.text
                })
        
        return rules 