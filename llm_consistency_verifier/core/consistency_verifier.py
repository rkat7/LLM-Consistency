import logging
import time
import hashlib
from typing import Dict, List, Optional, Any, Union
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
                logger.info(f"Cache hit (hits: {self.cache_hits}, misses: {self.cache_misses})")
                return self.cache[cache_key]
            else:
                self.cache_misses += 1
        
        start_time = time.time()
        logger.info(f"Starting consistency verification of text ({len(text)} chars)")
        
        # Extract logical rules from text using the appropriate method
        if self.use_llm_for_extraction:
            logical_rules = self.llm_interface.extract_logical_rules(text)
            logger.debug(f"Extracted {len(logical_rules)} logical rules using LLM")
        else:
            logical_rules = self.rule_extractor.extract_rules(text)
            logger.debug(f"Extracted {len(logical_rules)} logical rules using rule extractor")
        
        # No rules found
        if not logical_rules:
            logger.info("No logical rules found in text")
            result = VerificationResult(True, [])
            result.verification_time = time.time() - start_time
            
            # Add to cache if enabled
            if self.cache is not None:
                cache_key = self._get_cache_key(text)
                self.cache[cache_key] = result
                
            return result
        
        # Verify consistency of extracted rules
        verification_result = self.verification_engine.verify(logical_rules)
        
        # Add to cache if enabled
        if self.cache is not None:
            cache_key = self._get_cache_key(text)
            self.cache[cache_key] = verification_result
        
        # Log verification time
        total_time = time.time() - start_time
        logger.info(f"Total verification completed in {total_time:.2f}s: {'Consistent' if verification_result.is_consistent else 'Inconsistent'}")
        
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
        # First verify to find inconsistencies
        verification_result = self.verify(text)
        
        # If already consistent, return original text
        if verification_result.is_consistent:
            logger.info("Text is already consistent, no repair needed")
            return text
        
        # Use environment variable or config for max repair attempts
        if max_repair_attempts is None:
            max_repair_attempts = Config.MAX_REPAIR_ATTEMPTS
            
        logger.info(f"Repairing text with {len(verification_result.inconsistencies)} inconsistencies (max attempts: {max_repair_attempts})")
        
        # Use LLM to repair inconsistencies
        current_text = text
        best_text = text
        lowest_inconsistency_count = len(verification_result.inconsistencies)
        
        for attempt in range(max_repair_attempts):
            # Generate repaired text
            repaired_text = self.llm_interface.repair_inconsistencies(
                current_text, 
                verification_result.inconsistencies
            )
            
            # Verify the repaired text
            verification_result = self.verify(repaired_text)
            
            # If consistent, return the repaired text
            if verification_result.is_consistent:
                logger.info(f"Repaired text is consistent (attempt {attempt + 1}/{max_repair_attempts})")
                return repaired_text
            
            # Track the best version so far
            current_inconsistency_count = len(verification_result.inconsistencies)
            if current_inconsistency_count < lowest_inconsistency_count:
                lowest_inconsistency_count = current_inconsistency_count
                best_text = repaired_text
                logger.info(f"Repair attempt {attempt + 1}/{max_repair_attempts} reduced inconsistencies from {len(self.verify(current_text).inconsistencies)} to {current_inconsistency_count}")
                current_text = repaired_text
            else:
                logger.info(f"Repair attempt {attempt + 1}/{max_repair_attempts} did not improve text, keeping best version")
        
        # Return the best version we could achieve
        logger.warning(f"Could not fully repair text after {max_repair_attempts} attempts, returning best version with {lowest_inconsistency_count} inconsistencies")
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