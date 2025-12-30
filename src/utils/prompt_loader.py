"""Prompt loader utility for LLM interactions.

Loads prompts from YAML files in the prompts/ directory.
"""

from pathlib import Path
from typing import Any

import yaml
from loguru import logger


class PromptLoader:
    """Load and format prompts from YAML files."""

    def __init__(self, prompts_dir: Path | str = "prompts"):
        """Initialize prompt loader.

        Args:
            prompts_dir: Directory containing prompt YAML files
        """
        self.prompts_dir = Path(prompts_dir)
        if not self.prompts_dir.exists():
            raise FileNotFoundError(f"Prompts directory not found: {self.prompts_dir}")

    def load_prompt(self, step_name: str) -> dict[str, str]:
        """Load prompt template for a specific step.

        Args:
            step_name: Name of the step (e.g., "step3_clustering")

        Returns:
            Dictionary with 'system_prompt' and 'user_prompt' keys

        Raises:
            FileNotFoundError: If prompt file doesn't exist
            ValueError: If prompt file is invalid
        """
        prompt_file = self.prompts_dir / f"{step_name}.yaml"

        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

        try:
            with open(prompt_file, encoding="utf-8") as f:
                prompt_data = yaml.safe_load(f)

            if not isinstance(prompt_data, dict):
                raise ValueError(f"Invalid prompt file format: {prompt_file}")

            if "system_prompt" not in prompt_data or "user_prompt" not in prompt_data:
                raise ValueError(
                    f"Prompt file must contain 'system_prompt' and 'user_prompt': {prompt_file}"
                )

            logger.debug(f"Loaded prompt from {prompt_file}")
            return {
                "system_prompt": prompt_data["system_prompt"].strip(),
                "user_prompt": prompt_data["user_prompt"].strip(),
            }

        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse prompt YAML: {e}") from e

    def format_prompt(self, step_name: str, **kwargs: Any) -> str:
        """Load and format prompt with variables.

        Args:
            step_name: Name of the step (e.g., "step3_clustering")
            **kwargs: Variables to substitute in the prompt template

        Returns:
            Formatted complete prompt (system + user)
        """
        prompts = self.load_prompt(step_name)

        # Format user prompt with provided variables
        try:
            user_prompt_formatted = prompts["user_prompt"].format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required prompt variable: {e}") from e

        # Combine system and user prompts
        complete_prompt = f"{prompts['system_prompt']}\n\n{user_prompt_formatted}"

        logger.debug(f"Formatted prompt for {step_name} ({len(complete_prompt)} chars)")
        return complete_prompt

    def get_system_prompt(self, step_name: str) -> str:
        """Get only the system prompt for a step.

        Args:
            step_name: Name of the step

        Returns:
            System prompt text
        """
        prompts = self.load_prompt(step_name)
        return prompts["system_prompt"]

    def get_user_prompt_template(self, step_name: str) -> str:
        """Get only the user prompt template for a step.

        Args:
            step_name: Name of the step

        Returns:
            User prompt template (unformatted)
        """
        prompts = self.load_prompt(step_name)
        return prompts["user_prompt"]


# Singleton instance
_prompt_loader: PromptLoader | None = None


def get_prompt_loader() -> PromptLoader:
    """Get global PromptLoader instance.

    Returns:
        Singleton PromptLoader instance
    """
    global _prompt_loader
    if _prompt_loader is None:
        _prompt_loader = PromptLoader()
    return _prompt_loader
