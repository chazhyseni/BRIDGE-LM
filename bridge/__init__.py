"""
BRIDGE-LM - Biological Regulatory Integration & Detection through Genomic Language Models

A hierarchical transformer-based framework that orchestrates multiple genomic 
language models to detect regulatory elements with both genome-wide context 
and nucleotide-level precision.
"""

__version__ = "1.0.0"
__author__ = "Chaz Hyseni"

from .orchestrator import WorkflowOrchestrator

__all__ = ["WorkflowOrchestrator"]
