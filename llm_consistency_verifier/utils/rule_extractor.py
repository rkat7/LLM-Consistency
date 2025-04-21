import re
import logging
import json
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
    """Base class for extracting logical rules from text."""
    
    def __init__(self):
        """Initialize the rule extractor."""
        logger.info("Initializing rule extractor with pattern-based techniques")
        # Precompile regex patterns for better performance
        self.universal_pattern = re.compile(r'(?:all|every|each)\s+(.+?)\s+(?:are|is|have|has|must|will|can)\s+(.+?)(?:$|\.|\,)', re.IGNORECASE)
        self.existential_pattern = re.compile(r'(?:some|there\s+exists?|a few|many|most|several)\s+(.+?)\s+(?:are|is|have|has|can|do|will)\s+(.+?)(?:$|\.|\,)', re.IGNORECASE)
        self.negation_pattern = re.compile(r'(?:no|not all|none of the)\s+(.+?)\s+(?:are|is|have|has|can|will)\s+(.+?)(?:$|\.|\,)', re.IGNORECASE)
        self.assertion_pattern = re.compile(r'([^\.]+?)\s+(?:is|are|has|have|can|will|must|should)\s+([^\.]+?)(?:$|\.|\,)', re.IGNORECASE)
        self.implication_pattern = re.compile(r'(?:if|when|whenever)\s+(.+?)\s+(?:then|,)\s+(.+?)(?:$|\.|\,)', re.IGNORECASE)
        self.not_pattern = re.compile(r'([^\.]+?)\s+(?:is not|are not|isn\'t|aren\'t|cannot|can\'t|won\'t|will not|doesn\'t|does not|don\'t|do not)\s+([^\.]+?)(?:$|\.|\,)', re.IGNORECASE)
        self.equality_pattern = re.compile(r"([^\.]+?)\s+(?:is|are)\s+(?:the\s+)?same\s+(?:as)?\s+([^\.]+?)(?:$|\.|\,)", re.IGNORECASE)
    
    def extract_rules(self, text: str) -> List[Dict[str, Any]]:
        """Extract logical rules from text using pattern matching."""
        if not text:
            return []
            
        # Create a clear log entry for the extraction process
        logger.info(f"[FLOW:EXTRACTION:DETAIL] Starting rule extraction for text of length {len(text)}")
        
        # Process the text to prepare for extraction
        sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
        logger.info(f"[FLOW:EXTRACTION:DETAIL] Split text into {len(sentences)} sentences")
        
        rules = []
        
        # Process each sentence in parallel for large texts
        if len(sentences) > 10:
            # Process in batches for large texts
            batch_size = 10
            logger.info(f"[FLOW:EXTRACTION:DETAIL] Processing text in batches of {batch_size} sentences")
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i+batch_size]
                batch_rules = self._process_batch(batch)
                rules.extend(batch_rules)
                logger.info(f"[FLOW:EXTRACTION:DETAIL] Processed batch {i//batch_size + 1}, found {len(batch_rules)} rules")
        else:
            # Process sequentially for small texts
            logger.info(f"[FLOW:EXTRACTION:DETAIL] Processing text sequentially")
            for sentence in sentences:
                sentence_rules = self._extract_rules_from_sentence(sentence)
                rules.extend(sentence_rules)
                if sentence_rules:
                    logger.debug(f"[FLOW:EXTRACTION:DETAIL] Found {len(sentence_rules)} rules in sentence: '{sentence}'")
        
        # Remove duplicates preserving order
        unique_rules = []
        seen = set()
        for rule in rules:
            rule_key = self._get_rule_key(rule)
            if rule_key not in seen:
                seen.add(rule_key)
                unique_rules.append(rule)
        
        logger.info(f"[FLOW:EXTRACTION:DETAIL] Extracted {len(unique_rules)} unique logical rules total")
        
        return unique_rules
    
    def _process_batch(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """Process a batch of sentences in parallel."""
        batch_rules = []
        for sentence in sentences:
            batch_rules.extend(self._extract_rules_from_sentence(sentence))
        return batch_rules
    
    def _get_rule_key(self, rule: Dict[str, Any]) -> str:
        """Generate a unique key for a rule to help with deduplication."""
        if rule['type'] == 'implication':
            return f"implication:{rule.get('antecedent', '')}:{rule.get('consequent', '')}"
        else:
            return f"{rule['type']}:{rule.get('statement', '')}"
    
    def _extract_rules_from_sentence(self, sentence: str) -> List[Dict[str, Any]]:
        """Extract logical rules from a single sentence."""
        rules = []
        
        # Check for universal statements (All X are Y)
        universal_matches = self.universal_pattern.findall(sentence)
        for match in universal_matches:
            subject, predicate = match
            rules.append({
                "type": "universal",
                "statement": f"All {subject.strip()} are {predicate.strip()}",
                "original_text": sentence
            })
        
        # Check for existential statements (Some X are Y)
        existential_matches = self.existential_pattern.findall(sentence)
        for match in existential_matches:
            subject, predicate = match
            rules.append({
                "type": "existential",
                "statement": f"Some {subject.strip()} are {predicate.strip()}",
                "original_text": sentence
            })
        
        # Check for negation statements (No X are Y)
        negation_matches = self.negation_pattern.findall(sentence)
        for match in negation_matches:
            subject, predicate = match
            rules.append({
                "type": "negation",
                "statement": f"{subject.strip()} are {predicate.strip()}",  # Store positive form
                "original_text": sentence
            })
        
        # Check for implication statements (If X then Y)
        implication_matches = self.implication_pattern.findall(sentence)
        for match in implication_matches:
            antecedent, consequent = match
            rules.append({
                "type": "implication",
                "antecedent": antecedent.strip(),
                "consequent": consequent.strip(),
                "original_text": sentence
            })
        
        # Check for "not" statements (X is not Y)
        not_matches = self.not_pattern.findall(sentence)
        for match in not_matches:
            subject, predicate = match
            # Store the positive form for negation
            rules.append({
                "type": "negation",
                "statement": f"{subject.strip()} is {predicate.strip()}",  # Store positive form
                "original_text": sentence
            })
        
        # Check for equality statements (X is same as Y)
        equality_matches = self.equality_pattern.findall(sentence)
        for match in equality_matches:
            left, right = match
            rules.append({
                "type": "assertion",
                "statement": f"{left.strip()} is {right.strip()}",
                "original_text": sentence,
                "subtype": "equality"  # Mark as equality for special handling
            })
        
        # Check for regular assertions (X is Y) - do last to avoid overlap
        if not rules:  # Only if no other rule type was found
            assertion_matches = self.assertion_pattern.findall(sentence)
            for match in assertion_matches:
                subject, predicate = match
                # Skip if the predicate includes a negation
                if not any(neg in predicate.lower() for neg in ['not', "n't"]):
                    rules.append({
                        "type": "assertion",
                        "statement": f"{subject.strip()} is {predicate.strip()}",
                        "original_text": sentence
                    })
        
        return rules


class SpacyRuleExtractor(RuleExtractor):
    """Extracts logical rules using spaCy NLP for better entity recognition."""
    
    def __init__(self):
        """Initialize the SpaCy-based rule extractor."""
        super().__init__()
        
        try:
            # Try to load a larger model first
            import spacy
            try:
                self.nlp = spacy.load("en_core_web_lg")
                logger.info("Initialized spaCy large model (en_core_web_lg)")
            except OSError:
                logger.warning("Large model not found, falling back to en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Initialized spaCy small model (en_core_web_sm)")
                
            # Set up coreference resolution (placeholder for now)
            self.setup_coref()
            
        except (ImportError, OSError) as e:
            logger.error(f"Failed to initialize spaCy: {str(e)}")
            raise
    
    def setup_coref(self):
        """Set up coreference resolution."""
        # This is a placeholder for future implementation
        logger.info("Coreference resolution model loading is currently a placeholder.")
        self.coref_model = None
    
    def extract_rules(self, text: str) -> List[Dict[str, Any]]:
        """Extract logical rules from text using spaCy."""
        if not text:
            return []
        
        logger.info(f"[FLOW:EXTRACTION:DETAIL:SPACY] Starting spaCy-based rule extraction for text length {len(text)}")
        
        # For very large texts, split into manageable chunks
        if len(text) > 10000:
            logger.info(f"[FLOW:EXTRACTION:DETAIL:SPACY] Text is very large ({len(text)} chars), processing in chunks")
            return self._process_large_text(text)
        
        # Use spaCy for efficient processing
        doc = self.nlp(text)
        # Convert the sentence generator to a list to get the count
        sentences = list(doc.sents)
        logger.info(f"[FLOW:EXTRACTION:DETAIL:SPACY] Processed text with spaCy, found {len(sentences)} sentences")
        
        rules = []
        
        # Process each sentence to extract logical rules
        for i, sent in enumerate(sentences):
            sent_rules = self._extract_rules_from_spacy_sent(sent)
            rules.extend(sent_rules)
            if sent_rules:
                logger.debug(f"[FLOW:EXTRACTION:DETAIL:SPACY] Found {len(sent_rules)} rules in sentence {i+1}")
        
        # Enhanced entity resolution using spaCy's entity recognition
        resolved_rules = self._resolve_entities(rules, doc)
        
        # Remove duplicates while preserving order
        unique_rules = []
        seen = set()
        for rule in resolved_rules:
            rule_key = super()._get_rule_key(rule)
            if rule_key not in seen:
                seen.add(rule_key)
                unique_rules.append(rule)
        
        logger.info(f"[FLOW:EXTRACTION:DETAIL:SPACY] Extracted {len(unique_rules)} unique rules after entity resolution")
        
        return unique_rules
    
    def _process_large_text(self, text: str) -> List[Dict[str, Any]]:
        """Process very large texts by splitting into manageable chunks."""
        # Split into paragraphs or chunks of reasonable size
        chunks = []
        current_chunk = ""
        for paragraph in text.split("\n"):
            if len(current_chunk) + len(paragraph) < 5000:
                current_chunk += paragraph + "\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = paragraph + "\n"
        if current_chunk:
            chunks.append(current_chunk)
        
        logger.info(f"[FLOW:EXTRACTION:DETAIL:SPACY] Split large text into {len(chunks)} chunks")
        
        # Process each chunk
        all_rules = []
        for i, chunk in enumerate(chunks):
            logger.info(f"[FLOW:EXTRACTION:DETAIL:SPACY] Processing chunk {i+1}/{len(chunks)}")
            # Use the regular extract method on each chunk
            doc = self.nlp(chunk)
            chunk_rules = []
            for sent in doc.sents:
                chunk_rules.extend(self._extract_rules_from_spacy_sent(sent))
            all_rules.extend(chunk_rules)
        
        # Resolve entities across all rules
        resolved_rules = self._resolve_entities(all_rules, self.nlp(" ".join([r.get("original_text", "") for r in all_rules])))
        
        # Remove duplicates
        unique_rules = []
        seen = set()
        for rule in resolved_rules:
            rule_key = super()._get_rule_key(rule)
            if rule_key not in seen:
                seen.add(rule_key)
                unique_rules.append(rule)
        
        return unique_rules
    
    def _extract_rules_from_spacy_sent(self, sent) -> List[Dict[str, Any]]:
        """Extract rules from a spaCy sentence."""
        # Convert to a plain string for regex matching
        sent_text = sent.text
        
        # Use the parent class's sentence extraction method
        rules = super()._extract_rules_from_sentence(sent_text)
        
        # Additional spaCy-specific enhancements can be added here
        # For example, using dependency parsing to identify subjects and objects more accurately
        
        return rules
    
    def _resolve_entities(self, rules: List[Dict[str, Any]], doc) -> List[Dict[str, Any]]:
        """Resolve and normalize entities using spaCy's NER."""
        resolved_rules = []
        
        # Create an entity map to normalize entity references
        entity_map = {}
        for ent in doc.ents:
            entity_map[ent.text.lower()] = ent.text
        
        # Process each rule to normalize entities
        for rule in rules:
            new_rule = rule.copy()
            
            if rule["type"] == "implication":
                new_rule["antecedent"] = self._normalize_text(rule["antecedent"], entity_map)
                new_rule["consequent"] = self._normalize_text(rule["consequent"], entity_map)
            elif "statement" in rule:
                new_rule["statement"] = self._normalize_text(rule["statement"], entity_map)
            
            resolved_rules.append(new_rule)
        
        return resolved_rules
    
    def _normalize_text(self, text: str, entity_map: Dict[str, str]) -> str:
        """Normalize text using entity mapping."""
        # Check if the full text is in the entity map
        if text.lower() in entity_map:
            return entity_map[text.lower()]
        
        # Otherwise, keep the original text
        return text 