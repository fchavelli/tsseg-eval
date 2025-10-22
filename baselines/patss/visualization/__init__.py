"""
The visualization module offers functions to visually inspect the pattern-
based embedding and the obtained semantic segmentation.
"""
from .visualize_segmentation import visualize_segmentation, update_runtime_configuration_parameters

__all__ = [
    'visualize_segmentation',
    'update_runtime_configuration_parameters',
]
