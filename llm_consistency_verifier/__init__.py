"""
LLM Consistency Verifier
========================

A Python library for verifying logical consistency of LLM-generated text.
"""

# Ensure imports use relative paths
from .core.consistency_verifier import ConsistencyVerifier
from .core.verification_engine import VerificationEngine, VerificationResult
from .utils.llm_interface import LLMInterface
from .utils.text_processor import TextProcessor

__version__ = "0.1.0"
__all__ = [
    "ConsistencyVerifier",
    "VerificationEngine",
    "VerificationResult",
    "LLMInterface",
    "TextProcessor"
]

# Create singleton instance for easy import
_default_verifier = ConsistencyVerifier()
verify = _default_verifier.verify
repair = _default_verifier.repair 