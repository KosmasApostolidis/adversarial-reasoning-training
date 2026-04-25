"""CLI helpers shared by the ``scripts/*.py`` entry points.

These wrap the small amount of config-loading and runtime-setup
boilerplate that previously appeared inline in every script. Library
code should keep using the corresponding utilities directly
(``utils.seed.seed_everything`` etc.) — this module exists for the
script layer.
"""

from .config import load_yaml
from .runtime import setup_device, setup_run_dir, setup_seed

__all__ = ["load_yaml", "setup_device", "setup_run_dir", "setup_seed"]
