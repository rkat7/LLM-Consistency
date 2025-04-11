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
        self.lock = False
    
    def wait_if_needed(self):
        """Wait if we've hit the rate limit."""
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
                raise ValueError("OpenAI API key is required but not provided")
            return OpenAI(api_key=Config.OPENAI_API_KEY)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    def extract_logical_rules(self, text: str) -> List[Dict[str, Any]]:
        """Use the LLM to extract logical rules from text."""
        prompt = Config.EXTRACTION_PROMPT_TEMPLATE.format(text=text)
        
        response = self._generate_response(prompt)
        rules = self._parse_rules_from_response(response)
        
        logger.debug(f"Extracted {len(rules)} rules from text")
        return rules
    
    def repair_inconsistencies(self, text: str, inconsistencies: List[str]) -> str:
        """Use the LLM to repair inconsistencies in the text."""
        inconsistencies_text = "\n".join(f"- {inc}" for inc in inconsistencies)
        prompt = Config.REPAIR_PROMPT_TEMPLATE.format(
            text=text,
            inconsistencies=inconsistencies_text
        )
        
        repaired_text = self._generate_response(prompt)
        logger.debug(f"Generated repaired text: {repaired_text[:100]}...")
        
        return repaired_text
    
    def _generate_response(self, prompt: str) -> str:
        """Generate a response from the LLM."""
        if self.provider == "openai":
            return self._generate_openai_response(prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    def _generate_openai_response(self, prompt: str) -> str:
        """Generate a response using OpenAI API with retry logic and caching."""
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
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a logical reasoning assistant that helps extract formal logical rules from text and identify inconsistencies."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,  # Low temperature for more deterministic responses
                    max_tokens=1000,
                    timeout=Config.TIMEOUT_SECONDS
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
                
            except (APIError, APIConnectionError) as e:
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
        # This is a simplified parsing logic
        # A full implementation would use more robust parsing
        
        rules = []
        lines = response.strip().split("\n")
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith("-") or line.startswith("*") or ":" in line[:10]:
                # Skip formatting markers and empty lines
                continue
                
            # Attempt to convert the line into a rule
            try:
                # Check for common logical structures
                if "implies" in line.lower() or "→" in line or "->" in line:
                    parts = line.replace("→", "->").split("->")
                    if len(parts) == 2:
                        rule = {
                            "type": "implication",
                            "antecedent": parts[0].strip(),
                            "consequent": parts[1].strip(),
                            "original_text": line
                        }
                        rules.append(rule)
                elif "if" in line.lower() and "then" in line.lower():
                    parts = line.lower().split("then")
                    if len(parts) == 2:
                        antecedent = parts[0].replace("if", "", 1).strip()
                        rule = {
                            "type": "implication",
                            "antecedent": antecedent,
                            "consequent": parts[1].strip(),
                            "original_text": line
                        }
                        rules.append(rule)
                elif "all" in line.lower():
                    rule = {
                        "type": "universal",
                        "statement": line,
                        "original_text": line
                    }
                    rules.append(rule)
                elif "not" in line.lower() or "¬" in line:
                    rule = {
                        "type": "negation",
                        "statement": line,
                        "original_text": line
                    }
                    rules.append(rule)
                else:
                    rule = {
                        "type": "assertion",
                        "statement": line,
                        "original_text": line
                    }
                    rules.append(rule)
            except Exception as e:
                logger.warning(f"Failed to parse rule from line: {line}, error: {str(e)}")
        
        return rules
