"""General-purpose helper utilities."""

import logging
import os
from pathlib import Path


def configure_logging(level: str = "INFO") -> None:
    """Configure root logger with a standard format.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
    """
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
    )


def require_env(name: str) -> str:
    """Return the value of an environment variable, raising if absent.

    Args:
        name: Environment variable name.

    Returns:
        The variable's string value.

    Raises:
        EnvironmentError: If the variable is not set or is empty.
    """
    value = os.environ.get(name, "").strip()
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{name}' is not set. "
            "Copy .env.example to .env and fill in the values."
        )
    return value


def ensure_dir(path: Path) -> Path:
    """Create a directory and all parents if they do not exist.

    Args:
        path: Directory path to create.

    Returns:
        The same path after creation.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_latest_checkpoint(checkpoint_dir: Path) -> Path:
    """Find the most recent checkpoint file in a directory.

    Args:
        checkpoint_dir: Directory to search for checkpoints.

    Returns:
        Path to the checkpoint index file with the highest step number.

    Raises:
        FileNotFoundError: If no checkpoint files are found.
    """
    checkpoints = sorted(checkpoint_dir.glob("ckpt-*.index"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return checkpoints[-1].with_suffix("")
