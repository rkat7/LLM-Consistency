import logging
import time
from typing import Dict, List, Any, Optional, Union

from ..config.config import Config
from .consistency_verifier import ConsistencyVerifier
from .verification_engine import VerificationResult
from .enhanced_verification_engine import EnhancedVerificationEngine
from ..utils.advanced_rule_extractor import AdvancedRuleExtractor
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

class EnhancedConsistencyVerifier(ConsistencyVerifier):
    """
    Enhanced consistency verifier with support for complex logical reasoning.
    
    This extends the base ConsistencyVerifier to handle:
    - Complex conditionals with nested quantifiers
    - Temporal and modal logic
    - Hierarchical ontology relationships
    - More robust rule extraction
    """
    
    def __init__(self, 
                 llm_provider: str = Config.LLM_PROVIDER,
                 llm_model: str = Config.LLM_MODEL,
                 solver_type: str = Config.SOLVER_TYPE,
                 use_llm_for_extraction: bool = True):
        """
        Initialize the enhanced consistency verifier.
        
        Args:
            llm_provider: The LLM provider to use
            llm_model: The LLM model to use
            solver_type: The solver type to use
            use_llm_for_extraction: Whether to use LLM for rule extraction
        """
        # Initialize parent class
        super().__init__(llm_provider, llm_model, solver_type, use_llm_for_extraction)
        
        # Initialize enhanced components
        self.advanced_extractor = AdvancedRuleExtractor(self.llm_interface)
        self.ontology_manager = OntologyManager()
        self.enhanced_engine = EnhancedVerificationEngine(solver_type)
        
        # Override rule extractor if using LLM
        if use_llm_for_extraction:
            self.rule_extractor = self.advanced_extractor
        
        logger.info(f"EnhancedConsistencyVerifier initialized with LLM: {llm_provider}/{llm_model}, Solver: {solver_type}")
    
    def verify(self, text: str) -> VerificationResult:
        """
        Verify logical consistency with enhanced capabilities.
        
        Args:
            text: The text to verify
            
        Returns:
            VerificationResult: The verification result
        """
        # Check cache if enabled
        if self.cache is not None:
            cache_key = self._get_cache_key(text)
            if cache_key in self.cache:
                self.cache_hits += 1
                logger.info(f"[FLOW:ENHANCED] Cache hit (hits: {self.cache_hits}, misses: {self.cache_misses})")
                return self.cache[cache_key]
            else:
                self.cache_misses += 1
                logger.info(f"[FLOW:ENHANCED] Cache miss (hits: {self.cache_hits}, misses: {self.cache_misses})")
        
        start_time = time.time()
        
        # ------ STEP 1: RECEIVE TEXT INPUT ------
        logger.info(f"[FLOW:ENHANCED:INPUT] Starting enhanced verification of text ({len(text)} chars)")
        logger.debug(f"[FLOW:ENHANCED:INPUT] Text input: {text[:200]}...")
        
        # Extract original statements from the text
        original_statements = [s.strip() for s in text.split('.') if s.strip()]
        
        # ------ STEP 2: ENHANCED RULE EXTRACTION ------
        logger.info(f"[FLOW:ENHANCED:EXTRACTION] Extracting logical rules with advanced extractor")
        
        # Extract logical rules using advanced extractor
        try:
            logical_rules = self.advanced_extractor.extract_rules(text)
            logger.info(f"[FLOW:ENHANCED:EXTRACTION] Extracted {len(logical_rules)} logical rules using advanced extractor")
        except Exception as e:
            logger.warning(f"[FLOW:ENHANCED:EXTRACTION] Advanced extractor failed: {str(e)}. Falling back to standard extraction.")
            # Fall back to standard extraction
            if hasattr(self, 'use_llm_for_extraction') and self.use_llm_for_extraction:
                logical_rules = self.llm_interface.extract_logical_rules(text)
            else:
                logical_rules = self.rule_extractor.extract_rules(text)
            logger.info(f"[FLOW:ENHANCED:EXTRACTION] Extracted {len(logical_rules)} logical rules using fallback extractor")
        
        # No rules found - try direct LLM extraction as a backup
        if len(logical_rules) < 2:
            logger.warning(f"[FLOW:ENHANCED:EXTRACTION] Only {len(logical_rules)} rules found, attempting LLM backup extraction")
            llm_rules = self.llm_interface.extract_logical_rules(text)
            logger.info(f"[FLOW:ENHANCED:EXTRACTION] LLM backup extraction found {len(llm_rules)} rules")
            
            # If LLM found more rules, use those instead
            if len(llm_rules) > len(logical_rules):
                logger.info(f"[FLOW:ENHANCED:EXTRACTION] Using {len(llm_rules)} LLM-extracted rules instead of {len(logical_rules)} pattern-extracted rules")
                logical_rules = llm_rules
        
        # ------ STEP 3: ENHANCED VERIFICATION ------
        logger.info(f"[FLOW:ENHANCED:VERIFICATION] Performing enhanced verification on {len(logical_rules)} rules")
        
        try:
            # Use enhanced verification engine
            verification_result = self.enhanced_engine.verify(logical_rules, original_statements)
            logger.info(f"[FLOW:ENHANCED:VERIFICATION] Enhanced verification complete")
        except Exception as e:
            logger.warning(f"[FLOW:ENHANCED:VERIFICATION] Enhanced verification failed: {str(e)}. Falling back to standard verification.")
            # Fall back to standard verification
            verification_result = super().verify(text)
        
        # Set verification time
        verification_result.verification_time = time.time() - start_time
        
        # Add to cache if enabled
        if self.cache is not None:
            cache_key = self._get_cache_key(text)
            self.cache[cache_key] = verification_result
        
        return verification_result
    
    def verify_enhanced(self, text: str) -> VerificationResult:
        """
        Explicitly use enhanced verification without fallback.
        
        This method skips the fallback to standard verification,
        useful when you specifically want to test the enhanced capabilities.
        
        Args:
            text: The text to verify
            
        Returns:
            VerificationResult: The verification result
        """
        start_time = time.time()
        
        # Extract original statements
        original_statements = [s.strip() for s in text.split('.') if s.strip()]
        
        # Extract rules using advanced extractor
        logical_rules = self.advanced_extractor.extract_rules(text)
        
        # Use enhanced verification engine directly
        verification_result = self.enhanced_engine.verify(logical_rules, original_statements)
        
        # Set verification time
        verification_result.verification_time = time.time() - start_time
        
        return verification_result
    
    def repair(self, text: str, max_repair_attempts: int = None) -> str:
        """
        Repair logical inconsistencies with enhanced capabilities.
        
        Args:
            text: The text to repair
            max_repair_attempts: Maximum number of repair attempts
            
        Returns:
            str: The repaired text
        """
        # Use enhanced verification to identify inconsistencies
        verification_result = self.verify(text)
        
        # If already consistent, return original text
        if verification_result.is_consistent:
            logger.info("[FLOW:ENHANCED:REPAIR] Text is already consistent, no repair needed")
            return text
        
        # Use environment variable or config for max repair attempts
        if max_repair_attempts is None:
            max_repair_attempts = Config.MAX_REPAIR_ATTEMPTS
            
        logger.info(f"[FLOW:ENHANCED:REPAIR] Preparing to repair text with {len(verification_result.inconsistencies)} inconsistencies (max attempts: {max_repair_attempts})")
        
        # Iterative repair process
        current_text = text
        best_text = text
        lowest_inconsistency_count = len(verification_result.inconsistencies)
        
        for attempt in range(max_repair_attempts):
            logger.info(f"[FLOW:ENHANCED:REPAIR] Starting repair attempt {attempt + 1}/{max_repair_attempts}")
            
            # Generate repaired text
            repaired_text = self.llm_interface.repair_inconsistencies(
                current_text, 
                verification_result.inconsistencies
            )
            
            # Verify the repaired text
            verification_result = self.verify(repaired_text)
            
            # If consistent, return the repaired text
            if verification_result.is_consistent:
                logger.info(f"[FLOW:ENHANCED:REPAIR] Repair successful on attempt {attempt + 1}/{max_repair_attempts}")
                return repaired_text
            
            # Track the best version so far
            current_inconsistency_count = len(verification_result.inconsistencies)
            if current_inconsistency_count < lowest_inconsistency_count:
                lowest_inconsistency_count = current_inconsistency_count
                best_text = repaired_text
                logger.info(f"[FLOW:ENHANCED:REPAIR] Repair improved: {lowest_inconsistency_count} inconsistencies remain")
                current_text = repaired_text
            else:
                logger.info(f"[FLOW:ENHANCED:REPAIR] Repair did not improve, keeping best version")
        
        # Return the best version we could achieve
        logger.warning(f"[FLOW:ENHANCED:REPAIR] Could not fully repair after {max_repair_attempts} attempts")
        return best_text 