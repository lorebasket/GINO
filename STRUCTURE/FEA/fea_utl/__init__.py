"""
FEA Utilities Package

This package contains utility modules for the Finite Element Analysis (FEA) system.
"""

# Make modules available at package level
from . import analysis
from . import modal_reduction

__all__ = ['analysis', 'modal_reduction']
