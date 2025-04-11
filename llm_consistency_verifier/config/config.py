import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration settings for the LLM Consistency Verifier."""
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # LLM Model Settings
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # openai, huggingface, etc.
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4")  # gpt-4, llama2, etc.
    
    # Verification Engine Settings
    SOLVER_TYPE = os.getenv("SOLVER_TYPE", "z3")  # z3, sympy, etc.
    MAX_VERIFICATION_ITERATIONS = int(os.getenv("MAX_VERIFICATION_ITERATIONS", "5"))
    TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "30"))
    
    # Rule Extraction Settings
    USE_LLM_FOR_EXTRACTION = os.getenv("USE_LLM_FOR_EXTRACTION", "False").lower() == "true"
    USE_SPACY_EXTRACTOR = os.getenv("USE_SPACY_EXTRACTOR", "True").lower() == "true"
    EXTRACTION_PROMPT_TEMPLATE = """
    Extract logical rules and facts from the following text. 
    Format each as a logical statement or predicate:
    
    Text: {text}
    
    Rules and Facts:
    """
    
    # Repair Settings
    MAX_REPAIR_ATTEMPTS = int(os.getenv("MAX_REPAIR_ATTEMPTS", "3"))
    REPAIR_PROMPT_TEMPLATE = """
    The following text contains logical inconsistencies:
    
    {text}
    
    The inconsistencies are:
    {inconsistencies}
    
    Please provide a corrected version that resolves these inconsistencies while preserving the original meaning as much as possible.
    """
    
    # Log Settings
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "logs/consistency_verifier.log")
    
    # Performance Settings
    CACHE_RESULTS = os.getenv("CACHE_RESULTS", "True").lower() == "true"
    PARALLEL_PROCESSING = os.getenv("PARALLEL_PROCESSING", "False").lower() == "true"

    @classmethod
    def validate(cls):
        """Validate the configuration settings."""
        if not cls.OPENAI_API_KEY and cls.LLM_PROVIDER == "openai":
            raise ValueError("OPENAI_API_KEY is required when using OpenAI as the LLM provider")
        
        if cls.SOLVER_TYPE not in ["z3", "sympy"]:
            raise ValueError(f"Unsupported solver type: {cls.SOLVER_TYPE}")
            
        return True
