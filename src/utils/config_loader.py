"""Configuration loading utilities."""

from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.models.config import FeedsConfig, PipelineConfig

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


def load_yaml_config[T: BaseModel](file_path: Path | str, model_class: type[T]) -> T:
    """
    Load and validate YAML configuration file.

    Args:
        file_path: Path to YAML configuration file
        model_class: Pydantic model class to validate against

    Returns:
        Validated configuration instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValidationError: If config doesn't match schema
        yaml.YAMLError: If YAML is malformed

    Examples:
        >>> from src.models.config import PipelineConfig
        >>> config = load_yaml_config("config/pipeline.yaml", PipelineConfig)
    """
    path = Path(file_path)

    if not path.exists():
        logger.error("Configuration file not found", path=str(path))
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        with path.open("r") as f:
            raw_config = yaml.safe_load(f)

        config = model_class.model_validate(raw_config)
        logger.info("Configuration loaded", path=str(path), model=model_class.__name__)
        return config

    except yaml.YAMLError as e:
        logger.error("Invalid YAML syntax", path=str(path), error=str(e))
        raise

    except ValidationError as e:
        logger.error("Configuration validation failed", path=str(path), error=str(e))
        raise


def load_feeds_config(file_path: Path | str = "config/feeds.yaml") -> "FeedsConfig":
    """
    Load feeds configuration.

    Args:
        file_path: Path to feeds.yaml file

    Returns:
        FeedsConfig instance
    """
    from src.models.config import FeedsConfig

    return load_yaml_config(file_path, FeedsConfig)


def load_pipeline_config(file_path: Path | str = "config/pipeline.yaml") -> "PipelineConfig":
    """
    Load pipeline configuration.

    Args:
        file_path: Path to pipeline.yaml file

    Returns:
        PipelineConfig instance
    """
    from src.models.config import PipelineConfig

    return load_yaml_config(file_path, PipelineConfig)
