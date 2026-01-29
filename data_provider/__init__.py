"""Data loading and dataset utilities.

This package is intentionally lightweight and reusable:
- io.py: read & align price/macro panels + compute log-returns
- mech_graph.py: build normalized mechanism adjacency from edge list
- dataset_gp.py: multi-task window dataset returning (x_seq, y_seq)
"""

from .io import load_panel_from_two_files
from .mech_graph import build_adjacency_from_edges
from .dataset_gp import GPMultiTaskDataset

__all__ = [
    "load_panel_from_two_files",
    "build_adjacency_from_edges",
    "GPMultiTaskDataset",
]
