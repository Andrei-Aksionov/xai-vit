"""When imported returns parsed config file in a form of a dictionary with 'dot access'."""
from pathlib import Path

from omegaconf import OmegaConf

config = OmegaConf.load(Path(__file__).parent.parent / "config/config.yaml")
