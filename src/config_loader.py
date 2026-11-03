
from pathlib import Path
import yaml
from dotenv import load_dotenv

# Root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_DIR = PROJECT_ROOT / "config"

# Load local environment variables
load_dotenv(PROJECT_ROOT / ".env")


def load_config(config_name: str) -> dict:
    """
    Load a YAML configuration file.

    Example:
        load_config("data")
        load_config("rag")
        load_config("finetuning")
    """

    config_path = CONFIG_DIR / f"{config_name}.yaml"

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open(
        "r", encoding="utf-8"
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Invalid YAML configuration: {config_path}"
        )

    return config
