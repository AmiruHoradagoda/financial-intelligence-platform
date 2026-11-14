"""Repository configuration and local environment loading."""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv


def find_project_root() -> Path:
    """Find the checkout from source or cwd; allow an installed package override."""
    override = os.getenv("FINANCIAL_INTELLIGENCE_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
        if not (root / "config").is_dir():
            raise FileNotFoundError(
                f"Configuration directory not found: {root / 'config'}"
            )
        return root

    for start in (Path(__file__).resolve().parent, Path.cwd()):
        for candidate in (start, *start.parents):
            if (candidate / "pyproject.toml").is_file() and (
                candidate / "config"
            ).is_dir():
                return candidate
    raise FileNotFoundError(
        "Project root not found; set FINANCIAL_INTELLIGENCE_ROOT to the checkout"
    )


PROJECT_ROOT = find_project_root()
CONFIG_DIR = PROJECT_ROOT / "config"
load_dotenv(PROJECT_ROOT / ".env")


def load_config(config_name: str) -> dict:
    """Load an existing YAML mapping without changing its keys."""
    config_path = CONFIG_DIR / f"{config_name}.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML configuration: {config_path}")
    return config
