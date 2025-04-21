import logging
import time
import hashlib
from typing import Dict, List, Optional, Any, Union, Tuple
from functools import lru_cache

from ..config.config import Config
from ..utils.llm_interface import LLMInterface
from ..utils.rule_extractor import RuleExtractor, SpacyRuleExtractor
from .verification_engine import VerificationEngine, VerificationResult

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

class ConsistencyVerifier:
    """
    Main class for verifying logical consistency of LLM outputs.
    Integrates rule extraction, verification, and repair.
    """
    
    def __init__(self, 
                 llm_provider: str = Config.LLM_PROVIDER,
                 llm_model: str = Config.LLM_MODEL,
                 solver_type: str = Config.SOLVER_TYPE,
                 use_llm_for_extraction: bool = False):
        """Initialize the consistency verifier."""
        self.llm_interface = LLMInterface(provider=llm_provider, model=llm_model)
        self.verification_engine = VerificationEngine(solver_type=solver_type)
        self.cache = {} if Config.CACHE_RESULTS else None
        self.cache_hits = 0
        self.cache_misses = 0
        self.use_llm_for_extraction = use_llm_for_extraction
        
        # Initialize the appropriate rule extractor
        try:
            self.rule_extractor = SpacyRuleExtractor()
            logger.info("Using SpacyRuleExtractor for rule extraction")
        except Exception as e:
            logger.warning(f"Failed to initialize SpacyRuleExtractor: {str(e)}")
            self.rule_extractor = RuleExtractor()
            logger.info("Using basic RuleExtractor for rule extraction")
            
        logger.info(f"ConsistencyVerifier initialized with LLM: {llm_provider}/{llm_model}, Solver: {solver_type}, Using LLM for extraction: {use_llm_for_extraction}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate a unique cache key based on the text content."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def verify(self, text: str) -> VerificationResult:
        """
        Verify the logical consistency of text.
        
        Args:
            text: The text to verify for logical consistency
            
        Returns:
            VerificationResult object containing consistency status and details
        """
        # Check cache if enabled
        if self.cache is not None:
            cache_key = self._get_cache_key(text)
            if cache_key in self.cache:
                self.cache_hits += 1
                logger.info(f"[FLOW] Cache hit (hits: {self.cache_hits}, misses: {self.cache_misses})")
                return self.cache[cache_key]
            else:
                self.cache_misses += 1
                logger.info(f"[FLOW] Cache miss (hits: {self.cache_hits}, misses: {self.cache_misses})")
        
        start_time = time.time()
        
        # ------ STEP 1: RECEIVE TEXT INPUT ------
        logger.info(f"[FLOW:INPUT] Starting consistency verification of text ({len(text)} chars)")
        logger.debug(f"[FLOW:INPUT] Text input: {text[:200]}...")
        
        # Extract original statements from the text
        original_statements = [s.strip() for s in text.split('.') if s.strip()]
        
        # ------ STEP 2: RULE EXTRACTION ------
        logger.info(f"[FLOW:EXTRACTION] Extracting logical rules from text")
        
        # Extract logical rules from text using the appropriate method
        if self.use_llm_for_extraction:
            logger.info(f"[FLOW:LLM] Extracting logical rules from text using LLM")
            logical_rules = self.llm_interface.extract_logical_rules(text)
            logger.info(f"[FLOW:EXTRACTION] Extracted {len(logical_rules)} logical rules using LLM")
        else:
            logger.info(f"[FLOW:EXTRACTION] Extracting logical rules from text")
            logical_rules = self.rule_extractor.extract_rules(text)
            logger.info(f"[FLOW:EXTRACTION] Extracted {len(logical_rules)} logical rules using pattern-based extractor")
        
        # No rules found - try direct LLM extraction as a backup if rules are few
        # This covers cases where the pattern matching fails but the text is inconsistent
        if len(logical_rules) < 2 and not self.use_llm_for_extraction:
            logger.warning(f"[FLOW:EXTRACTION] Only {len(logical_rules)} rules found, attempting LLM backup extraction")
            llm_rules = self.llm_interface.extract_logical_rules(text)
            logger.info(f"[FLOW:EXTRACTION] LLM backup extraction found {len(llm_rules)} rules")
            
            # If LLM found more rules, use those instead
            if len(llm_rules) > len(logical_rules):
                logger.info(f"[FLOW:EXTRACTION] Using {len(llm_rules)} LLM-extracted rules instead of {len(logical_rules)} pattern-extracted rules")
                logical_rules = llm_rules
        
        # Log extracted rules
        for i, rule in enumerate(logical_rules):
            if rule["type"] == "implication":
                rule_str = f"{rule['antecedent']} -> {rule['consequent']}"
            else:
                rule_str = rule.get("statement", "")
            logger.debug(f"[FLOW:EXTRACTION] Rule {i+1}: Type={rule['type']}, Content={rule_str}")
        
        # ------ STEP 3: FORMAL VERIFICATION ------
        logger.info(f"[FLOW:FORMALIZATION] Performing formal verification on {len(logical_rules)} rules")
            
        # Still no rules found? Handle empty case differently    
        if not logical_rules:
            logger.info("[FLOW:FORMALIZATION] No logical rules found in text, skipping verification")
            
            # Check if text is extremely short, then it can be considered consistent
            if len(text.strip()) < 10:
                result = VerificationResult(True, [])
            else:
                # For longer texts, absence of logical rules is suspicious
                # Indicate this isn't verified rather than claiming consistency
                result = VerificationResult(True, ["No logical rules were extracted from this text, consistency cannot be fully verified."])
            
            result.verification_time = time.time() - start_time
            
            # Add to cache if enabled
            if self.cache is not None:
                cache_key = self._get_cache_key(text)
                self.cache[cache_key] = result
                
            return result
        
        # Verify consistency of extracted rules, passing original statements for pattern matching
        verification_result = self.verification_engine.verify(logical_rules, original_statements)
        
        # ------ STEP 4: RESULTS ------
        if verification_result.is_consistent:
            logger.info("[FLOW:RESULTS] Verification result: CONSISTENT")
        else:
            logger.info(f"[FLOW:RESULTS] Verification result: INCONSISTENT with {len(verification_result.inconsistencies)} inconsistencies")
            for i, inconsistency in enumerate(verification_result.inconsistencies):
                logger.debug(f"[FLOW:RESULTS] Inconsistency {i+1}: {inconsistency}")
        
        # Add to cache if enabled
        if self.cache is not None:
            cache_key = self._get_cache_key(text)
            self.cache[cache_key] = verification_result
        
        # Log verification time
        total_time = time.time() - start_time
        logger.info(f"[FLOW:COMPLETE] Total verification completed in {total_time:.2f}s")
        
        return verification_result
    
    def repair(self, text: str, max_repair_attempts: int = None) -> str:
        """
        Repair logical inconsistencies in text.
        
        Args:
            text: The text containing logical inconsistencies
            max_repair_attempts: Override the default max repair attempts config
            
        Returns:
            Corrected text with inconsistencies resolved
        """
        # ------ STEP 1: VERIFY TO FIND INCONSISTENCIES ------
        logger.info(f"[FLOW:REPAIR:INPUT] First verifying text to identify inconsistencies")
        verification_result = self.verify(text)
        
        # If already consistent, return original text
        if verification_result.is_consistent:
            logger.info("[FLOW:REPAIR:INPUT] Text is already consistent, no repair needed")
            return text
        
        # ------ STEP 2: REPAIR PREPARATION ------
        # Use environment variable or config for max repair attempts
        if max_repair_attempts is None:
            max_repair_attempts = Config.MAX_REPAIR_ATTEMPTS
            
        logger.info(f"[FLOW:REPAIR:PLANNING] Preparing to repair text with {len(verification_result.inconsistencies)} inconsistencies (max attempts: {max_repair_attempts})")
        logger.debug(f"[FLOW:REPAIR:PLANNING] Original text: {text[:200]}...")
        
        for i, inconsistency in enumerate(verification_result.inconsistencies):
            logger.debug(f"[FLOW:REPAIR:PLANNING] Inconsistency {i+1}: {inconsistency}")
        
        # ------ STEP 3: ITERATIVE REPAIR ------
        # Use LLM to repair inconsistencies
        current_text = text
        best_text = text
        lowest_inconsistency_count = len(verification_result.inconsistencies)
        
        for attempt in range(max_repair_attempts):
            logger.info(f"[FLOW:REPAIR:ATTEMPT] Starting repair attempt {attempt + 1}/{max_repair_attempts}")
            
            # Generate repaired text
            logger.debug(f"[FLOW:REPAIR:LLM] Repairing {len(verification_result.inconsistencies)} inconsistencies")
            repaired_text = self.llm_interface.repair_inconsistencies(
                current_text, 
                verification_result.inconsistencies
            )
            logger.debug(f"[FLOW:REPAIR:LLM] Received repaired text: {repaired_text[:200]}...")
            
            # Verify the repaired text
            logger.info(f"[FLOW:REPAIR:EVALUATION] Verifying repair attempt {attempt + 1}")
            verification_result = self.verify(repaired_text)
            
            # If consistent, return the repaired text
            if verification_result.is_consistent:
                logger.info(f"[FLOW:REPAIR:EVALUATION] Repair successful! Text is now consistent (attempt {attempt + 1}/{max_repair_attempts})")
                return repaired_text
            
            # Track the best version so far
            current_inconsistency_count = len(verification_result.inconsistencies)
            if current_inconsistency_count < lowest_inconsistency_count:
                lowest_inconsistency_count = current_inconsistency_count
                best_text = repaired_text
                logger.info(f"[FLOW:REPAIR:EVALUATION] Repair attempt {attempt + 1}/{max_repair_attempts} reduced inconsistencies from {len(self.verify(current_text).inconsistencies)} to {current_inconsistency_count}")
                
                # Fix by handling the string processing outside the f-string
                remaining_inconsistencies = []
                for inc in verification_result.inconsistencies:
                    first_line = inc.split('\n')[0] if '\n' in inc else inc
                    remaining_inconsistencies.append(first_line)
                
                logger.debug(f"[FLOW:REPAIR:EVALUATION] Remaining inconsistencies: {', '.join(remaining_inconsistencies)}")
                current_text = repaired_text
            else:
                logger.info(f"[FLOW:REPAIR:EVALUATION] Repair attempt {attempt + 1}/{max_repair_attempts} did not improve text, keeping best version")
        
        # ------ STEP 4: FINALIZE REPAIR ------
        # Return the best version we could achieve
        logger.warning(f"[FLOW:REPAIR:EVALUATION] Could not fully repair text after {max_repair_attempts} attempts, returning best version with {lowest_inconsistency_count} inconsistencies")
        return best_text
    
    def explain_inconsistencies(self, verification_result: VerificationResult) -> str:
        """
        Generate a human-readable explanation of inconsistencies.
        
        Args:
            verification_result: The result of verification
            
        Returns:
            Human-readable explanation of inconsistencies
        """
        if verification_result.is_consistent:
            return "The text is logically consistent."
        
        explanation = "The following logical inconsistencies were detected:\n\n"
        
        for i, inconsistency in enumerate(verification_result.inconsistencies, 1):
            explanation += f"{i}. {inconsistency}\n"
        
        explanation += "\nTo resolve these inconsistencies, you could:\n"
        explanation += "- Clarify the scope of general statements\n"
        explanation += "- Remove or modify contradicting assertions\n"
        explanation += "- Add context or exceptions where needed\n"
        
        return explanation 