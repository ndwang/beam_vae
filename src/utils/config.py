"""Configuration loading and management utilities."""

import copy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries, with override taking precedence."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def parse_override(override_str: str) -> tuple[List[str], Any]:
    """Parse a CLI override string like 'model.latent_dim=128' or 'training.lr=1e-4'.

    Returns:
        Tuple of (key_path, value) where key_path is a list of nested keys.
    """
    if '=' not in override_str:
        raise ValueError(f"Override must be in format 'key=value', got: {override_str}")

    key, value_str = override_str.split('=', 1)
    key_path = key.split('.')

    # Try to parse value as YAML (handles int, float, bool, lists, etc.)
    try:
        value = yaml.safe_load(value_str)
    except yaml.YAMLError:
        value = value_str

    return key_path, value


def apply_override(config: Dict, key_path: List[str], value: Any) -> None:
    """Apply a single override to a config dict in place."""
    current = config
    for key in key_path[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[key_path[-1]] = value


def apply_overrides(config: Dict, overrides: List[str]) -> Dict:
    """Apply CLI overrides to config.

    Args:
        config: Base configuration dictionary.
        overrides: List of override strings like ['model.latent_dim=128', 'training.lr=1e-4'].

    Returns:
        Modified config dictionary.
    """
    config = copy.deepcopy(config)
    for override_str in overrides:
        key_path, value = parse_override(override_str)
        apply_override(config, key_path, value)
    return config


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    config_dir: Union[str, Path] = "configs",
    overrides: Optional[List[str]] = None,
    validate: bool = True,
) -> Dict[str, Any]:
    """Load and compose configuration from YAML files.

    Args:
        config_path: Path to main config file. Defaults to config_dir/default.yaml.
        config_dir: Base directory for config files.
        overrides: List of CLI override strings.
        validate: Whether to validate the config against the schema.

    Returns:
        Composed configuration dictionary.

    Raises:
        ConfigValidationError: If validate=True and config is invalid.
    """
    config_dir = Path(config_dir)

    if config_path is None:
        config_path = config_dir / "default.yaml"
    else:
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = config_dir / config_path

    # Load main config
    config = load_yaml(config_path)

    # Compose sub-configs
    composed = {}
    for key in ['model', 'training', 'data']:
        if key in config and isinstance(config[key], str):
            # It's a path reference, load the sub-config
            sub_path = config_dir / config[key]
            composed[key] = load_yaml(sub_path)
        elif key in config and isinstance(config[key], dict):
            # It's an inline config
            composed[key] = config[key]

    # Add non-composed keys
    for key, value in config.items():
        if key not in ['model', 'training', 'data']:
            composed[key] = value

    # Apply CLI overrides
    if overrides:
        composed = apply_overrides(composed, overrides)

    # Validate config against schema
    if validate:
        from .validation import validate_config
        composed = validate_config(composed)

    return composed


def save_config(config: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save configuration to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def config_to_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert flat config to model config format expected by VAE classes."""
    model_cfg = config.get('model', {})
    return {'model': model_cfg}


def generate_run_name(config: Dict[str, Any]) -> str:
    """Generate concise run name from key params + short timestamp.

    Format: latent{dim}_beta{beta}_{MMDD}_{HHMM}
    Example: latent64_beta1e-05_0126_1430
    """
    latent_dim = config.get('model', {}).get('latent_dim', 0)
    beta = config.get('training', {}).get('beta', 0)
    timestamp = datetime.now().strftime("%y%m%d_%H%M")

    return f"latent{latent_dim}_beta{beta}_{timestamp}"
