import os
import json
from typing import Dict, List, Optional, Union, Any
import logging
import requests
import time
import random
import hashlib
import pickle
from pathlib import Path
from openai import OpenAI, RateLimitError, APIError, APIConnectionError
from ..config.config import Config
import re
import threading  # Add threading import

# Ensure log directory exists
log_dir = os.path.dirname(Config.LOG_FILE)
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

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

class RateLimiter:
    """Simple rate limiter to prevent hitting API rate limits."""
    
    def __init__(self, calls_per_minute=20):
        self.calls_per_minute = calls_per_minute
        self.call_times = []
        self.lock = threading.Lock()  # Use threading.Lock instead of boolean
    
    def wait_if_needed(self):
        """Wait if we've hit the rate limit."""
        with self.lock:  # Use lock to ensure thread safety
            now = time.time()
            
            # Remove timestamps older than 1 minute
            self.call_times = [t for t in self.call_times if now - t < 60]
            
            # If we've hit the limit, wait
            if len(self.call_times) >= self.calls_per_minute:
                sleep_time = 61 - (now - self.call_times[0])
                if sleep_time > 0:
                    logger.info(f"Rate limit reached. Waiting for {sleep_time:.2f}s before next API call.")
                    time.sleep(sleep_time)
                    # Clear some old timestamps after waiting
                    self.call_times = self.call_times[len(self.call_times)//2:]
            
            # Add the current time to the list
            self.call_times.append(time.time())

class DiskCache:
    """Persistent disk-based cache for LLM responses."""
    
    def __init__(self, cache_dir=".cache"):
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True)
        logger.info(f"Using disk cache at {self.cache_dir}")
    
    def _get_cache_key(self, provider, model, content):
        """Generate a unique cache key based on the prompt, provider, and model."""
        key_material = f"{provider}:{model}:{content}"
        return hashlib.md5(key_material.encode()).hexdigest()
    
    def get(self, provider, model, content):
        """Get cached response if it exists."""
        cache_key = self._get_cache_key(provider, model, content)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    cached_data = pickle.load(f)
                logger.debug(f"Cache hit for {provider}/{model}")
                return cached_data
            except Exception as e:
                logger.warning(f"Failed to load cache: {str(e)}")
        
        return None
    
    def set(self, provider, model, content, response):
        """Cache the response."""
        cache_key = self._get_cache_key(provider, model, content)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(response, f)
            logger.debug(f"Cached response for {provider}/{model}")
        except Exception as e:
            logger.warning(f"Failed to cache response: {str(e)}")


class LLMInterface:
    """Interface for interacting with large language models."""
    
    # Class-level rate limiter to ensure rate limits apply across all instances
    _rate_limiter = RateLimiter(calls_per_minute=int(os.getenv("API_CALLS_PER_MINUTE", "20")))
    _disk_cache = DiskCache() if os.getenv("USE_DISK_CACHE", "True").lower() == "true" else None
    
    def __init__(self, provider: str = Config.LLM_PROVIDER, model: str = Config.LLM_MODEL):
        """Initialize the LLM interface."""
        self.provider = provider
        self.model = model
        self.client = self._initialize_client()
        self.max_retries = int(os.getenv("MAX_API_RETRIES", "5"))
        logger.info(f"LLM Interface initialized with provider: {provider}, model: {model}")
    
    def _initialize_client(self) -> Any:
        """Initialize the appropriate client based on the provider."""
        if self.provider == "openai":
            if not Config.OPENAI_API_KEY:
                logger.error("OpenAI API key is required but not provided")
                raise ValueError("OpenAI API key is required but not provided. Please set OPENAI_API_KEY in your environment variables or .env file.")
            try:
                return OpenAI(api_key=Config.OPENAI_API_KEY)
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {str(e)}")
                raise
        else:
            logger.error(f"Unsupported LLM provider: {self.provider}")
            raise ValueError(f"Unsupported LLM provider: {self.provider}. Currently only 'openai' is supported.")
    
    def extract_logical_rules(self, text: str) -> List[Dict[str, Any]]:
        """Use the LLM to extract logical rules from text.
        
        Args:
            text: The text to extract logical rules from
            
        Returns:
            A list of dictionaries representing the extracted logical rules
        """
        # Check input
        if not text or not isinstance(text, str):
            logger.warning("Empty or invalid text provided for rule extraction")
            return []
            
        # Truncate very long text to prevent token limit issues
        max_text_length = 8000
        if len(text) > max_text_length:
            logger.warning(f"Text exceeds {max_text_length} characters, truncating for rule extraction")
            text = text[:max_text_length] + "..."
        
        # Enhanced extraction prompt for more reliable rule extraction
        prompt = f"""
Extract ALL logical rules and facts from the following text. Be thorough and INCLUDE EVERY LOGICAL STATEMENT.
Follow these guidelines EXACTLY:

1. FORMAT RULES:
   - Universal: "All X are Y"
   - Existential: "Some X are Y"
   - Implication: "If X then Y"
   - Negation: "X is not Y"
   - Assertion: "X is Y"

2. IDENTIFY EVERY rule, fact, or statement that could participate in logical reasoning.
3. Be comprehensive - extract ALL statements even if they seem simple or obvious.
4. Number each rule for clarity.
5. For the classic birds-penguin case: "All birds can fly. Penguins are birds. Penguins cannot fly.",
   you MUST extract THREE distinct rules:
   1. Universal: "All birds can fly"
   2. Assertion: "Penguins are birds"
   3. Negation: "Penguins cannot fly"

Text to analyze:
{text}

Rules and Facts (numbered):
"""
        
        logger.info(f"[FLOW:LLM:EXTRACTION] Starting LLM-based rule extraction for text: {text[:100]}...")
        
        try:
            response = self._generate_response(prompt)
            if not response:
                logger.warning("Empty response received from LLM for rule extraction")
                return []
                
            logger.debug(f"[FLOW:LLM:EXTRACTION] LLM extraction response: {response[:500]}...")
            
            rules = self._parse_rules_from_response(response)
            
            logger.info(f"[FLOW:LLM:EXTRACTION] Extracted {len(rules)} rules from text using LLM")
            for i, rule in enumerate(rules, 1):
                logger.debug(f"[FLOW:LLM:EXTRACTION] Rule {i}: Type={rule['type']}, Content={rule.get('statement', rule.get('antecedent', ''))}")
            
            # If no rules were extracted, try a fallback approach
            if not rules:
                logger.warning("No rules extracted from LLM response, trying fallback extraction")
                # Simple fallback extraction for common patterns
                if "All" in text and "are" in text:
                    parts = text.split(".")
                    for part in parts:
                        part = part.strip()
                        if "All" in part and "are" in part:
                            rules.append({
                                "type": "universal",
                                "statement": part,
                                "original_text": part
                            })
                        elif "is not" in part or "cannot" in part or "are not" in part:
                            rules.append({
                                "type": "negation",
                                "statement": part.replace("not ", "").replace("cannot", "can"),
                                "original_text": part
                            })
                        elif "is" in part or "are" in part:
                            rules.append({
                                "type": "assertion",
                                "statement": part,
                                "original_text": part
                            })
                logger.info(f"[FLOW:LLM:EXTRACTION] Fallback extraction found {len(rules)} rules")
            
            return rules
            
        except Exception as e:
            logger.error(f"[FLOW:LLM:EXTRACTION] Error extracting logical rules: {str(e)}")
            # Return empty list rather than crashing
            return []
    
    def repair_inconsistencies(self, text: str, inconsistencies: List[str]) -> str:
        """Use the LLM to repair inconsistencies in the text.
        
        Args:
            text: The original text that contains inconsistencies
            inconsistencies: List of identified inconsistencies to repair
            
        Returns:
            The repaired text with inconsistencies fixed
        """
        # Check inputs
        if not text or not isinstance(text, str):
            logger.warning("Empty or invalid text provided for repair")
            return text
            
        if not inconsistencies or not isinstance(inconsistencies, list):
            logger.warning("No inconsistencies provided for repair")
            return text
        
        try:
            # Format inconsistencies as a bulleted list
            inconsistencies_text = "\n".join(f"- {inc}" for inc in inconsistencies)
            
            # Truncate very long text to prevent token limit issues
            max_text_length = 8000
            if len(text) > max_text_length:
                logger.warning(f"Text exceeds {max_text_length} characters, truncating for repair")
                text = text[:max_text_length] + "..."
            
            prompt = Config.REPAIR_PROMPT_TEMPLATE.format(
                text=text,
                inconsistencies=inconsistencies_text
            )
            
            logger.info(f"[FLOW:LLM:REPAIR] Repairing {len(inconsistencies)} inconsistencies in text: {text[:100]}...")
            repaired_text = self._generate_response(prompt)
            
            if not repaired_text:
                logger.warning("Empty response received from LLM for inconsistency repair")
                return text
                
            logger.debug(f"[FLOW:LLM:REPAIR] Generated repaired text: {repaired_text[:100]}...")
            
            return repaired_text
            
        except Exception as e:
            logger.error(f"[FLOW:LLM:REPAIR] Error repairing inconsistencies: {str(e)}")
            # Return original text rather than crashing
            return text
    
    def _generate_response(self, prompt: str) -> str:
        """Generate a response from the LLM."""
        if self.provider == "openai":
            return self._generate_openai_response(prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    def _generate_openai_response(self, prompt: str) -> str:
        """Generate a response using OpenAI API with retry logic and caching.
        
        Args:
            prompt: The input prompt to generate a response for
            
        Returns:
            The generated text response
            
        Raises:
            Exception: If the API calls repeatedly fail after retries
        """
        # Check disk cache first if enabled
        if self._disk_cache:
            cached_response = self._disk_cache.get(self.provider, self.model, prompt)
            if cached_response:
                return cached_response
        
        retry_count = 0
        base_delay = 1  # Initial delay in seconds
        
        while retry_count <= self.max_retries:
            try:
                # Wait if we've hit the rate limit
                self._rate_limiter.wait_if_needed()
                
                # Set a reasonable timeout
                timeout = Config.TIMEOUT_SECONDS
                if timeout <= 0:
                    # Default to 60 seconds if timeout is invalid
                    timeout = 60
                    logger.warning(f"Invalid timeout value: {Config.TIMEOUT_SECONDS}, using default 60 seconds")
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a logical reasoning assistant that helps extract formal logical rules from text and identify inconsistencies."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,  # Low temperature for more deterministic responses
                    max_tokens=1000,
                    timeout=timeout
                )
                
                if not response.choices:
                    logger.error("No response generated from OpenAI")
                    return ""
                
                content = response.choices[0].message.content.strip()
                
                # Cache the response if disk cache is enabled
                if self._disk_cache:
                    self._disk_cache.set(self.provider, self.model, prompt, content)
                
                return content
                
            except RateLimitError as e:
                # Handle rate limit error with exponential backoff
                retry_count += 1
                if retry_count > self.max_retries:
                    logger.error(f"Max retries exceeded for rate limit error: {str(e)}")
                    raise
                
                # Calculate backoff delay with jitter
                delay = base_delay * (2 ** (retry_count - 1)) + random.uniform(0, 1)
                logger.warning(f"Rate limit error encountered, retrying in {delay:.2f}s (attempt {retry_count}/{self.max_retries})")
                time.sleep(delay)
                
            except (APIError, APIConnectionError, TimeoutError) as e:
                # Handle other API errors with exponential backoff
                retry_count += 1
                if retry_count > self.max_retries:
                    logger.error(f"Max retries exceeded for API error: {str(e)}")
                    raise
                
                # Calculate backoff delay with jitter
                delay = base_delay * (2 ** (retry_count - 1)) + random.uniform(0, 1)
                logger.warning(f"API error encountered: {str(e)}, retrying in {delay:.2f}s (attempt {retry_count}/{self.max_retries})")
                time.sleep(delay)
                
            except Exception as e:
                logger.error(f"Error generating response from OpenAI: {str(e)}")
                raise
        
        raise Exception(f"Failed to generate response after {self.max_retries} retries")
    
    def _parse_rules_from_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse logical rules from the LLM response."""
        rules = []
        lines = response.strip().split("\n")
        
        # Process each line that looks like a rule
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and obvious formatting
            if not line or line.lower() == "extracted rules:" or line.lower() == "rules and facts:":
                continue
                
            # Remove numbering, bullets and other formatting
            line = re.sub(r'^\d+[\.\)]\s*', '', line)  # Remove "1.", "2.", "3)." etc.
            line = re.sub(r'^[-*•]\s*', '', line)  # Remove bullet points
            
            # Skip section headers and other non-rule text
            if ":" in line[:15] and not re.search(r'if|implies|then', line[:15].lower()):
                continue
                
            # Process by matching different rule formats
            try:
                # Universal statements (All X are Y)
                if re.match(r'^(?:universal|all|every|each)\b', line.lower()):
                    # Improved pattern to catch more variations of universal statements
                    match = re.search(r'(?:all|every|each)\s+(.+?)\s+(?:are|is|have|has|must|should|can|will|do|does)\s+(.+?)(?:$|\.|\,)', line.lower())
                    if match:
                        subject = match.group(1).strip()
                        predicate = match.group(2).strip()
                        rules.append({
                            "type": "universal",
                            "statement": f"All {subject} are {predicate}",
                            "original_text": line
                        })
                        continue
                        
                # Existential statements (Some X are Y)
                elif re.match(r'^(?:existential|some|there\s+exists?|at\s+least\s+one)\b', line.lower()):
                    # Improved pattern for existential statements
                    match = re.search(r'(?:some|there\s+exists?|at\s+least\s+one)\s+(.+?)\s+(?:are|is|have|has|can|will|do|does)\s+(.+?)(?:$|\.|\,)', line.lower())
                    if match:
                        subject = match.group(1).strip()
                        predicate = match.group(2).strip()
                        rules.append({
                            "type": "existential",
                            "statement": f"Some {subject} are {predicate}",
                            "original_text": line
                        })
                        continue
                
                # Negation statements (X is not Y)
                elif re.match(r'^(?:negation|not)\b', line.lower()) or 'not' in line.lower():
                    # Explicit negation format "Not(P)"
                    not_match = re.search(r'not\s*[\(\{\[](.+?)[\)\}\]]', line.lower(), re.IGNORECASE)
                    if not_match:
                        statement = not_match.group(1).strip()
                        rules.append({
                            "type": "negation",
                            "statement": statement,  # Store positive form
                            "original_text": line
                        })
                        continue
                    
                    # X is not Y format - improved pattern to catch more variations
                    not_match = re.search(r'(.+?)\s+(?:is not|are not|cannot|can\'t|isn\'t|aren\'t|don\'t|do not|does not|doesn\'t|will not|won\'t|never)\s+(.+?)(?:$|\.|\,)', line.lower())
                    if not_match:
                        subject = not_match.group(1).strip()
                        predicate = not_match.group(2).strip()
                        # Store the positive form for consistency checking
                        positive_form = f"{subject} is {predicate}"
                        rules.append({
                            "type": "negation",
                            "statement": positive_form,  # Store positive form
                            "original_text": line
                        })
                        continue
                    
                # Implication statements (If X then Y)
                elif re.match(r'^(?:implication|if)\b', line.lower()) or 'implies' in line.lower() or '->' in line or '→' in line:
                    # If X then Y format
                    if_then_match = re.search(r'if\s+(.+?)\s+then\s+(.+?)(?:$|\.|\,)', line.lower())
                    if if_then_match:
                        antecedent = if_then_match.group(1).strip()
                        consequent = if_then_match.group(2).strip()
                        rules.append({
                            "type": "implication",
                            "antecedent": antecedent,
                            "consequent": consequent,
                            "original_text": line
                        })
                        continue
                    
                    # X implies Y or X -> Y format
                    implies_match = re.search(r'(.+?)\s+(?:implies|->|→|⟹|⇒)\s+(.+?)(?:$|\.|\,)', line)
                    if implies_match:
                        antecedent = implies_match.group(1).strip()
                        consequent = implies_match.group(2).strip()
                        rules.append({
                            "type": "implication",
                            "antecedent": antecedent,
                            "consequent": consequent,
                            "original_text": line
                        })
                        continue
                    
                    # When X, Y format (another implication form)
                    when_then_match = re.search(r'when\s+(.+?)\s*(?:,|\:)\s*(.+?)(?:$|\.|\,)', line.lower())
                    if when_then_match:
                        antecedent = when_then_match.group(1).strip()
                        consequent = when_then_match.group(2).strip()
                        rules.append({
                            "type": "implication",
                            "antecedent": antecedent,
                            "consequent": consequent,
                            "original_text": line
                        })
                        continue
                
                # Simple assertion statements (X is Y)
                assertion_match = re.search(r'(.+?)\s+(?:is|are|has|have|can|must|will|should)\s+(.+?)(?:$|\.|\,)', line.lower())
                if assertion_match:
                    subject = assertion_match.group(1).strip()
                    predicate = assertion_match.group(2).strip()
                    # Skip if the subject is too generic or a stopword
                    if len(subject.split()) == 1 and subject.lower() in ['this', 'that', 'it', 'they', 'he', 'she']:
                        continue
                    rules.append({
                        "type": "assertion",
                        "statement": f"{subject} is {predicate}",
                        "original_text": line
                    })
                    continue
                
                # If we get here, try to use the whole line as a general assertion
                if len(line.split()) >= 3 and not line.lower().startswith(('note:', 'rule:', 'example:')):
                    rules.append({
                        "type": "assertion",
                        "statement": line,
                        "original_text": line
                    })
                
            except Exception as e:
                logger.warning(f"Error parsing rule from line '{line}': {str(e)}")
                continue
        
        # Filter out exact duplicates
        unique_rules = []
        seen_rule_strings = set()
        
        for rule in rules:
            rule_str = f"{rule['type']}:{rule.get('statement', '')}{rule.get('antecedent', '')}{rule.get('consequent', '')}"
            if rule_str not in seen_rule_strings:
                seen_rule_strings.add(rule_str)
                unique_rules.append(rule)
        
        return unique_rules
