import json
import logging
from typing import List, Dict, Any, Optional
import re

from ..config.config import Config
from ..utils.rule_extractor import RuleExtractor
from ..utils.llm_interface import LLMInterface

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

class AdvancedRuleExtractor(RuleExtractor):
    """
    Enhanced rule extractor with support for complex logical structures.
    
    This extends the base RuleExtractor to handle:
    - Complex conditionals
    - Nested quantifiers
    - Temporal logic
    - Modal logic
    - Hierarchical ontology
    """
    
    def __init__(self, llm_interface: Optional[LLMInterface] = None):
        """Initialize the advanced rule extractor."""
        super().__init__()
        self.llm_interface = llm_interface or LLMInterface()
        logger.info("Advanced rule extractor initialized")
    
    def extract_rules(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract logical rules from text with support for complex structures.
        
        Args:
            text: The text to extract rules from
            
        Returns:
            A list of dictionaries representing the extracted rules
        """
        logger.info(f"[FLOW:ADVANCED_EXTRACTION] Starting advanced rule extraction for text ({len(text)} chars)")
        
        # First try the LLM-based structured extraction
        structured_rules = self.extract_structured_rules(text)
        
        if structured_rules:
            logger.info(f"[FLOW:ADVANCED_EXTRACTION] Successfully extracted {len(structured_rules)} structured rules")
            return structured_rules
        
        # Fall back to pattern-based extraction if LLM extraction fails
        logger.info("[FLOW:ADVANCED_EXTRACTION] Falling back to pattern-based extraction")
        return super().extract_rules(text)
    
    def extract_structured_rules(self, text: str) -> List[Dict[str, Any]]:
        """
        Use LLM to extract structured rules with advanced logical features.
        
        Args:
            text: The text to extract rules from
            
        Returns:
            A list of dictionaries representing the structured rules
        """
        # Define the structured extraction prompt
        structured_prompt = f"""
Parse the following text into formal logical structures. For EACH logical statement:

1. IDENTIFICATION:
   - Type: universal, existential, implication, assertion, negation
   - Format: Subject-Predicate-Object or Antecedent-Consequent for implications

2. ADVANCED FEATURES (when present):
   - Nested quantifiers (e.g., "All X such that there exists a Y...")
   - Temporal context (e.g., "always", "sometimes", "before", "after")
   - Modal operators (e.g., "necessarily", "possibly", "must", "should")
   - Ontological relationships (e.g., "is-a", "part-of", "instance-of")

3. ENTITY RECOGNITION:
   - Subject/object entities (e.g., "birds", "penguins")
   - Properties/attributes (e.g., "can fly", "have wings")
   - Relations between entities (e.g., "bigger than", "parent of")

TEXT TO ANALYZE:
{text}

Return your analysis as a JSON array with the following schema:
[
  {{
    "type": "universal|existential|implication|assertion|negation",
    "subject": "entity or concept",
    "predicate": "relation or property",
    "object": "entity, value or concept",
    "original_text": "exact text segment from the input",
    "nested_quantifiers": ["any nested quantifiers"],
    "temporal_context": "temporal information if present",
    "modal_context": "modal information if present",
    "ontological_relation": "is-a|part-of|instance-of if applicable"
  }}
]

IMPORTANT: Return ONLY the JSON array, with no additional text before or after.
"""
        
        try:
            # Generate structured response from LLM
            raw_response = self.llm_interface._generate_response(structured_prompt)
            
            # Ensure we have a valid JSON response
            cleaned_response = self._clean_json_response(raw_response)
            structured_data = json.loads(cleaned_response)
            
            # Convert to our internal rule format
            rules = self._convert_structured_data_to_rules(structured_data)
            
            return rules
            
        except Exception as e:
            logger.error(f"[FLOW:ADVANCED_EXTRACTION] Error in structured extraction: {str(e)}")
            return []
    
    def _clean_json_response(self, response: str) -> str:
        """
        Clean the response to ensure it contains valid JSON.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Cleaned JSON string
        """
        # Remove any text before the first [
        if '[' in response:
            start_idx = response.find('[')
            response = response[start_idx:]
        
        # Remove any text after the last ]
        if ']' in response:
            end_idx = response.rfind(']')
            response = response[:end_idx+1]
        
        # Handle common JSON formatting issues
        response = response.replace('```json', '').replace('```', '')
        
        return response.strip()
    
    def _convert_structured_data_to_rules(self, structured_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert structured data from LLM to our internal rule format.
        
        Args:
            structured_data: Structured data from LLM
            
        Returns:
            List of rules in our internal format
        """
        rules = []
        
        for item in structured_data:
            rule_type = item.get('type', '').lower()
            original_text = item.get('original_text', '')
            
            if not rule_type or not original_text:
                continue
            
            # Create base rule
            rule = {
                "type": rule_type,
                "original_text": original_text
            }
            
            # Add basic properties based on rule type
            if rule_type == 'implication':
                # For implications, we need antecedent and consequent
                antecedent = f"{item.get('subject', '')} {item.get('predicate', '')}"
                consequent = item.get('object', '')
                rule["antecedent"] = antecedent.strip()
                rule["consequent"] = consequent.strip()
            else:
                # For other types, construct a statement
                subject = item.get('subject', '')
                predicate = item.get('predicate', '')
                object_value = item.get('object', '')
                
                if rule_type == 'universal':
                    statement = f"All {subject} {predicate} {object_value}"
                elif rule_type == 'existential':
                    statement = f"Some {subject} {predicate} {object_value}"
                elif rule_type == 'negation':
                    statement = f"{subject} {predicate} {object_value}"
                else:  # assertion
                    statement = f"{subject} {predicate} {object_value}"
                
                rule["statement"] = statement.strip()
            
            # Add advanced properties if present
            if 'nested_quantifiers' in item and item['nested_quantifiers']:
                rule["nested_quantifiers"] = item["nested_quantifiers"]
            
            if 'temporal_context' in item and item['temporal_context']:
                rule["temporal_context"] = item["temporal_context"]
            
            if 'modal_context' in item and item['modal_context']:
                rule["modal_context"] = item["modal_context"]
            
            if 'ontological_relation' in item and item['ontological_relation']:
                rule["ontological_relation"] = item["ontological_relation"]
            
            rules.append(rule)
        
        return rules 