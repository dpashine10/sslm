"""Data acquisition, preparation, tokenization, and loading utilities.

The modules in this package are intentionally usable both as Python APIs and
as command-line programs.  See each module's ``--help`` output for the
corresponding command-line contract.
"""

__all__ = ["RandomWindowDataset"]


def __getattr__(name: str):
    """Load the optional PyTorch-backed dataset class on demand."""

    if name == "RandomWindowDataset":
        from .dataset import RandomWindowDataset

        return RandomWindowDataset
    raise AttributeError(name)
