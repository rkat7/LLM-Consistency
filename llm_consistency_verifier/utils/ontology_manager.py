import logging
from typing import Dict, Set, List, Tuple, Any, Optional
from collections import defaultdict

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

class OntologyManager:
    """
    Manages hierarchical relationships between entities and their properties.
    
    This handles:
    - Hierarchical class relationships (is-a)
    - Part-whole relationships (part-of)
    - Instance-class relationships (instance-of)
    - Property inheritance across hierarchies
    """
    
    def __init__(self):
        """Initialize the ontology manager."""
        # Hierarchical relationships
        self.is_a_relations = defaultdict(set)  # child -> {parents}
        self.part_of_relations = defaultdict(set)  # part -> {wholes}
        self.instance_of_relations = defaultdict(set)  # instance -> {classes}
        
        # Property management
        self.property_inheritance = defaultdict(dict)  # entity -> {property: value}
        self.property_exceptions = defaultdict(dict)  # entity -> {property: value}
        
        # Track all entities and properties
        self.all_entities = set()
        self.all_properties = set()
        
        logger.info("OntologyManager initialized")
    
    def add_entity(self, entity: str, parent: str = None, relationship: str = "is-a") -> None:
        """
        Add an entity to the ontology.
        
        Args:
            entity: The entity to add
            parent: Optional parent entity
            relationship: Relationship type ("is-a", "part-of", "instance-of")
        """
        self.all_entities.add(entity)
        
        # If parent is provided, add the relationship
        if parent is not None:
            if relationship == "is-a":
                self.add_is_a(entity, parent)
            elif relationship == "part-of":
                self.add_part_of(entity, parent)
            elif relationship == "instance-of":
                self.add_instance_of(entity, parent)
        
        logger.debug(f"Added entity {entity}")
    
    def add_property(self, property_name: str) -> None:
        """
        Add a property to the ontology.
        
        Args:
            property_name: The name of the property
        """
        self.all_properties.add(property_name)
        logger.debug(f"Added property {property_name}")
    
    def add_property_to_entity(self, entity: str, property_name: str, value: Any = True) -> None:
        """
        Add a property to an entity and propagate to children.
        
        Args:
            entity: The entity to add the property to
            property_name: The name of the property
            value: The value of the property
        """
        self.add_entity(entity)
        self.add_property(property_name)
        self.property_inheritance[entity][property_name] = value
        
        # Propagate to all children in is-a hierarchy
        self._propagate_property(entity, property_name, value)
        
        logger.debug(f"Added property {property_name}={value} to entity {entity}")
    
    def add_property_exception(self, entity: str, property_name: str, value: Any = False) -> None:
        """
        Add a property exception to an entity.
        
        This overrides inherited properties in specific cases.
        
        Args:
            entity: The entity to add the exception to
            property_name: The name of the property
            value: The value of the property
        """
        self.add_entity(entity)
        self.add_property(property_name)
        self.property_exceptions[entity][property_name] = value
        
        logger.debug(f"Added property exception {property_name}={value} to entity {entity}")
    
    def add_is_a(self, child: str, parent: str) -> None:
        """
        Add an 'is-a' relationship between entities.
        
        Args:
            child: The child entity (e.g., 'dog')
            parent: The parent entity (e.g., 'animal')
        """
        self.add_entity(child)
        self.add_entity(parent)
        self.is_a_relations[child].add(parent)
        
        # Propagate properties from parent to child
        for prop, value in self.property_inheritance.get(parent, {}).items():
            if prop not in self.property_exceptions.get(child, {}):
                self.property_inheritance.setdefault(child, {})[prop] = value
        
        logger.debug(f"Added is-a relationship: {child} is-a {parent}")
    
    def add_part_of(self, part: str, whole: str) -> None:
        """
        Add a 'part-of' relationship between entities.
        
        Args:
            part: The part entity (e.g., 'wheel')
            whole: The whole entity (e.g., 'car')
        """
        self.add_entity(part)
        self.add_entity(whole)
        self.part_of_relations[part].add(whole)
        
        logger.debug(f"Added part-of relationship: {part} part-of {whole}")
    
    def add_instance_of(self, instance: str, class_name: str) -> None:
        """
        Add an 'instance-of' relationship between entities.
        
        Args:
            instance: The instance entity (e.g., 'Fido')
            class_name: The class entity (e.g., 'dog')
        """
        self.add_entity(instance)
        self.add_entity(class_name)
        self.instance_of_relations[instance].add(class_name)
        
        # Propagate properties from class to instance
        for prop, value in self.property_inheritance[class_name].items():
            if prop not in self.property_exceptions[instance]:
                self.property_inheritance[instance][prop] = value
        
        logger.debug(f"Added instance-of relationship: {instance} instance-of {class_name}")
    
    def get_property(self, entity: str, property_name: str) -> Optional[Any]:
        """
        Get a property value for an entity.
        
        Args:
            entity: The entity to get the property for
            property_name: The name of the property
            
        Returns:
            The property value or None if not found
        """
        # Check for exception first
        if property_name in self.property_exceptions[entity]:
            return self.property_exceptions[entity][property_name]
        
        # Check direct property
        if property_name in self.property_inheritance[entity]:
            return self.property_inheritance[entity][property_name]
        
        # Look through instance-of relationships
        for class_name in self.instance_of_relations[entity]:
            value = self.get_property(class_name, property_name)
            if value is not None:
                return value
        
        # Look through is-a relationships
        for parent in self.is_a_relations[entity]:
            value = self.get_property(parent, property_name)
            if value is not None:
                return value
        
        return None
    
    def get_all_parents(self, entity: str) -> Set[str]:
        """
        Get all parents of an entity in the is-a hierarchy.
        
        Args:
            entity: The entity to get parents for
            
        Returns:
            Set of all parent entities
        """
        result = set()
        self._collect_all_parents(entity, result)
        return result
    
    def _collect_all_parents(self, entity: str, result: Set[str]) -> None:
        """
        Recursively collect all parents of an entity.
        
        Args:
            entity: The entity to collect parents for
            result: Set to store results in
        """
        for parent in self.is_a_relations[entity]:
            result.add(parent)
            self._collect_all_parents(parent, result)
    
    def get_all_classes(self, instance: str) -> Set[str]:
        """
        Get all classes that an instance belongs to.
        
        Args:
            instance: The instance to get classes for
            
        Returns:
            Set of all class entities
        """
        result = set()
        
        # Direct classes
        result.update(self.instance_of_relations[instance])
        
        # Parent classes of each direct class
        for class_name in self.instance_of_relations[instance]:
            result.update(self.get_all_parents(class_name))
        
        return result
    
    def _propagate_property(self, entity: str, property_name: str, value: Any) -> None:
        """
        Recursively propagate a property to all children in is-a hierarchy.
        
        Args:
            entity: The entity with the property
            property_name: The name of the property
            value: The value to propagate
        """
        # Find all children in is-a hierarchy
        for child in [c for c, parents in self.is_a_relations.items() if entity in parents]:
            # Only propagate if child doesn't have an exception
            if property_name not in self.property_exceptions[child]:
                self.property_inheritance[child][property_name] = value
                # Continue propagating down the hierarchy
                self._propagate_property(child, property_name, value)
        
        # Find all instances of this entity
        for instance in [i for i, classes in self.instance_of_relations.items() if entity in classes]:
            # Only propagate if instance doesn't have an exception
            if property_name not in self.property_exceptions[instance]:
                self.property_inheritance[instance][property_name] = value
    
    def process_rule(self, rule: Dict[str, Any]) -> None:
        """
        Process a rule and update the ontology accordingly.
        
        Args:
            rule: The rule to process
        """
        rule_type = rule.get('type', '').lower()
        
        # Handle different rule types
        if rule_type == 'universal':
            # "All X are Y" -> Add is-a relationship
            if 'statement' in rule:
                statement = rule['statement'].lower()
                if statement.startswith('all ') and ' are ' in statement:
                    parts = statement.split(' are ')
                    child = parts[0].replace('all ', '').strip()
                    parent = parts[1].strip()
                    self.add_is_a(child, parent)
            
        elif rule_type == 'assertion':
            # "X is a Y" -> Add instance-of relationship
            if 'statement' in rule:
                statement = rule['statement'].lower()
                if ' is a ' in statement:
                    parts = statement.split(' is a ')
                    instance = parts[0].strip()
                    class_name = parts[1].strip()
                    self.add_instance_of(instance, class_name)
        
        # Handle ontological relations explicitly marked
        if 'ontological_relation' in rule:
            relation = rule.get('ontological_relation', '').lower()
            subject = rule.get('subject', '').lower()
            object_value = rule.get('object', '').lower()
            
            if relation == 'is-a':
                self.add_is_a(subject, object_value)
            elif relation == 'part-of':
                self.add_part_of(subject, object_value)
            elif relation == 'instance-of':
                self.add_instance_of(subject, object_value)
        
        # Handle property assignment
        if rule_type in ['assertion', 'universal'] and 'statement' in rule:
            statement = rule['statement'].lower()
            
            # Check for property statements like "X has Y" or "X can Y"
            if ' has ' in statement:
                parts = statement.split(' has ')
                entity = parts[0].strip()
                if entity.startswith('all '):
                    entity = entity.replace('all ', '').strip()
                property_value = parts[1].strip()
                self.add_property_to_entity(entity, f"has_{property_value}")
            
            elif ' can ' in statement:
                parts = statement.split(' can ')
                entity = parts[0].strip()
                if entity.startswith('all '):
                    entity = entity.replace('all ', '').strip()
                ability = parts[1].strip()
                self.add_property_to_entity(entity, f"can_{ability}")
        
        # Handle property negation
        if rule_type == 'negation' and 'statement' in rule:
            statement = rule['statement'].lower()
            original_text = rule.get('original_text', '').lower()
            
            if ' cannot ' in original_text:
                parts = original_text.split(' cannot ')
                entity = parts[0].strip()
                ability = parts[1].strip()
                self.add_property_exception(entity, f"can_{ability}", False)
            
            elif ' does not have ' in original_text or ' doesn\'t have ' in original_text:
                # Split on either variant
                if ' does not have ' in original_text:
                    parts = original_text.split(' does not have ')
                else:
                    parts = original_text.split(' doesn\'t have ')
                
                entity = parts[0].strip()
                property_value = parts[1].strip()
                self.add_property_exception(entity, f"has_{property_value}", False)
        
        logger.debug(f"Processed rule: {rule_type} - {rule.get('original_text', '')}")
    
    def check_consistency(self) -> List[str]:
        """
        Check for inconsistencies in the ontology.
        
        Returns:
            List of inconsistency messages
        """
        inconsistencies = []
        
        # Check for property value conflicts
        for entity in self.all_entities:
            for prop in self.all_properties:
                if prop in self.property_inheritance[entity] and prop in self.property_exceptions[entity]:
                    inherited_value = self.property_inheritance[entity][prop]
                    exception_value = self.property_exceptions[entity][prop]
                    
                    if inherited_value != exception_value:
                        # Find the source of the inherited property
                        sources = []
                        for parent in self.get_all_parents(entity):
                            if prop in self.property_inheritance[parent]:
                                sources.append(parent)
                        
                        for class_name in self.get_all_classes(entity):
                            if prop in self.property_inheritance[class_name]:
                                sources.append(class_name)
                        
                        source_str = ", ".join(sources)
                        inconsistencies.append(
                            f"Inconsistency: Entity '{entity}' has conflicting values for property '{prop}'. " + 
                            f"Inherited as '{inherited_value}' from [{source_str}], but explicitly set to '{exception_value}'."
                        )
        
        return inconsistencies 