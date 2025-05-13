import logging
import time
from typing import Dict, List, Any, Optional, Union

from ..config.config import Config
from .consistency_verifier import ConsistencyVerifier
from .verification_engine import VerificationResult
from .enhanced_verification_engine import EnhancedVerificationEngine
from ..utils.advanced_rule_extractor import AdvancedRuleExtractor
from ..utils.ontology_manager import OntologyManager
from ..utils.rule_extractor import SpacyRuleExtractor
from ..utils.llm_interface import LLMInterface
from .verification_engine import VerificationEngine

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
                 use_llm_for_extraction: bool = True,
                 use_caching: bool = True):
        """
        Initialize the enhanced consistency verifier.
        
        Args:
            llm_provider: The LLM provider to use
            llm_model: The LLM model to use
            solver_type: The solver type to use
            use_llm_for_extraction: Whether to use LLM for rule extraction
            use_caching: Whether to cache results
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
        
        # Initialize the LLM interface
        self.llm = LLMInterface(provider=llm_provider, model=llm_model)
        
        # Initialize the verification engine
        self.verification_engine = VerificationEngine(solver_type=solver_type)
        
        # Cache for verification results
        self.cache = {}
        self.use_caching = use_caching
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info(f"EnhancedConsistencyVerifier initialized with LLM: {llm_provider}/{llm_model}, Solver: {solver_type}")
    
    def verify(self, text: str, detail_level: str = 'minimal') -> VerificationResult:
        """
        Verify the logical consistency of a given text.
        
        Args:
            text: The text to verify
            detail_level: Level of detail in the result ('minimal', 'detailed', 'debug')
            
        Returns:
            VerificationResult: The verification result
        """
        # Check cache first
        if self.use_caching and text in self.cache:
            self.cache_hits += 1
            logger.info(f"[FLOW:ENHANCED] Cache hit (hits: {self.cache_hits}, misses: {self.cache_misses})")
            return self.cache[text]
        
        self.cache_misses += 1
        logger.info(f"[FLOW:ENHANCED] Cache miss (hits: {self.cache_hits}, misses: {self.cache_misses})")
        
        # Start verification process
        logger.info(f"[FLOW:ENHANCED:INPUT] Starting enhanced verification of text ({len(text)} chars)")
        logger.debug(f"[FLOW:ENHANCED:INPUT] Text input: {text[:100]}...")
        
        start_time = time.time()
        
        # Extract logical rules
        logger.info("[FLOW:ENHANCED:EXTRACTION] Extracting logical rules with advanced extractor")
        extracted_rules = self.advanced_extractor.extract_rules(text)
        logger.info(f"[FLOW:ENHANCED:EXTRACTION] Extracted {len(extracted_rules)} logical rules using advanced extractor")
        
        # Break text into statements
        statements = self._split_into_statements(text)
        
        # Verify the logical consistency
        logger.info(f"[FLOW:ENHANCED:VERIFICATION] Performing enhanced verification on {len(extracted_rules)} rules")
        verification_result = self.verification_engine.verify(extracted_rules, statements)
        
        # Calculate verification time
        verification_time = time.time() - start_time
        if hasattr(verification_result, 'verification_time'):
            verification_result.verification_time = verification_time
        
        logger.info("[FLOW:ENHANCED:VERIFICATION] Enhanced verification complete")
        
        # Store the details in the verification_result for debugging if needed
        if hasattr(verification_result, 'details'):
            # Add details based on detail level
            if detail_level in ['detailed', 'debug']:
                verification_result.details['extracted_rules'] = extracted_rules
                
            if detail_level == 'debug':
                verification_result.details['statements'] = statements
        
        # Cache result
        if self.use_caching:
            self.cache[text] = verification_result
        
        return verification_result
    
    def _split_into_statements(self, text: str) -> List[str]:
        """
        Split the text into individual statements.
        
        Args:
            text: The text to split
            
        Returns:
            List of individual statements
        """
        # Preprocess the text
        text = text.replace(';', '.')
        
        # Split by period followed by space or end of string
        statements = []
        for segment in text.split('.'):
            segment = segment.strip()
            if segment:
                statements.append(segment + '.')
        
        return statements
    
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