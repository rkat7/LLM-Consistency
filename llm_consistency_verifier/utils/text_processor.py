import re
import logging
from typing import List, Dict, Optional, Set, Tuple
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

class TextProcessor:
    """Utility class for text processing and analysis."""
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Preprocess text for logical rule extraction.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Normalize punctuation
        text = text.replace('...', '.')
        text = re.sub(r'\.+', '.', text)
        
        # Ensure proper spacing after punctuation
        text = re.sub(r'(\.)([A-Za-z])', r'\1 \2', text)
        
        logger.debug(f"Preprocessed text: {text[:100]}...")
        return text
    
    @staticmethod
    def extract_sentences(text: str) -> List[str]:
        """
        Extract sentences from text.
        
        Args:
            text: Text to extract sentences from
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        logger.debug(f"Extracted {len(sentences)} sentences")
        return sentences
    
    @staticmethod
    def identify_logical_patterns(text: str) -> Dict[str, List[str]]:
        """
        Identify common logical patterns in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of pattern types and matching sentences
        """
        patterns = {
            "universal": [],
            "existential": [],
            "conditional": [],
            "negation": [],
            "assertion": []
        }
        
        sentences = TextProcessor.extract_sentences(text)
        
        for sentence in sentences:
            # Check for universal statements (All, Every, etc.)
            if re.search(r'\b(all|every|any|each)\b', sentence.lower()):
                patterns["universal"].append(sentence)
            
            # Check for existential statements (Some, There exists, etc.)
            elif re.search(r'\b(some|there (is|are|exists))\b', sentence.lower()):
                patterns["existential"].append(sentence)
            
            # Check for conditional statements (If... then)
            elif re.search(r'\bif\b.*\bthen\b', sentence.lower()) or '→' in sentence or '->' in sentence:
                patterns["conditional"].append(sentence)
            
            # Check for negation (Not, No, Never, etc.)
            elif re.search(r'\b(not|no|never|cannot|isn\'t|aren\'t)\b', sentence.lower()):
                patterns["negation"].append(sentence)
            
            # Otherwise, treat as assertion
            else:
                patterns["assertion"].append(sentence)
        
        logger.debug(f"Identified logical patterns: {', '.join(f'{k}: {len(v)}' for k, v in patterns.items())}")
        return patterns
    
    @staticmethod
    def analyze_contradiction_potential(text: str) -> float:
        """
        Analyze the potential for contradictions in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Score between 0-1 indicating contradiction potential
        """
        patterns = TextProcessor.identify_logical_patterns(text)
        
        # Calculate basic metrics
        universal_count = len(patterns["universal"])
        negation_count = len(patterns["negation"])
        conditional_count = len(patterns["conditional"])
        total_sentences = sum(len(v) for v in patterns.values())
        
        if total_sentences == 0:
            return 0.0
        
        # Heuristic: texts with more universals and negations have higher contradiction potential
        contradiction_score = (0.4 * universal_count + 0.4 * negation_count + 0.2 * conditional_count) / max(total_sentences, 1)
        
        # Normalize to 0-1
        contradiction_score = min(contradiction_score * 2, 1.0)
        
        logger.debug(f"Contradiction potential score: {contradiction_score:.2f}")
        return contradiction_score
    
    @staticmethod
    def extract_entities(text: str) -> Set[str]:
        """
        Extract potential logical entities from text.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Set of entity strings
        """
        # This is a simplified implementation
        # A full implementation would use NLP tools for entity extraction
        
        # Extract nouns using simple regex
        # This is very simplistic and would be replaced with proper NLP
        noun_pattern = r'\b([A-Z][a-z]+|[a-z]+)\b'
        potential_nouns = set(re.findall(noun_pattern, text))
        
        # Filter common words (very basic)
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'while', 'when'}
        entities = {word for word in potential_nouns if word.lower() not in stopwords and len(word) > 2}
        
        logger.debug(f"Extracted {len(entities)} potential entities")
        return entities 