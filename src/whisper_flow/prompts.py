"""Prompt management functionality for whisper-flow."""

import re
from typing import Any

from .config import Config
from .system import SystemManager


class PromptManager:
    """Manages prompt selection and template processing."""

    def __init__(self, config: Config, system_manager: SystemManager):
        """Initialize prompt manager.

        Args:
            config: Configuration object
            system_manager: System manager for window detection

        """
        self.config = config
        self.system_manager = system_manager

    def choose_prompt(
        self,
        window_title: str | None = None,
        mode: str = "default",
    ) -> str:
        """Choose appropriate prompt based on window context.

        Args:
            window_title: Current window title (auto-detected if None)
            mode: Prompt mode ('default' or 'dictation')

        Returns:
            Selected prompt template

        """
        if window_title is None:
            window_title = self.system_manager.get_active_window_title() or ""

        prompt_rules = self.config.get_prompts_config(mode)

        for rule in prompt_rules:
            pattern = rule.get("match", "")
            if pattern and re.search(pattern, window_title, re.IGNORECASE):
                return rule.get("prompt", "{{text}}")

        # Return default prompt if no match found
        return "{{text}}"

    def process_prompt(self, prompt_template: str, transcript: str) -> str:
        """Process prompt template with transcript.

        Args:
            prompt_template: Prompt template with placeholders
            transcript: Transcribed text to insert

        Returns:
            Processed prompt ready for completion

        """
        if not transcript.strip():
            return ""

        # Replace the main placeholder
        processed = prompt_template.replace("{{text}}", transcript)

        # Handle additional placeholders
        processed = self._replace_dynamic_placeholders(processed)

        return processed

    def _replace_dynamic_placeholders(self, prompt: str) -> str:
        """Replace dynamic placeholders in prompt.

        Args:
            prompt: Prompt with potential dynamic placeholders

        Returns:
            Prompt with dynamic placeholders replaced

        """
        from datetime import datetime

        # Time-based placeholders
        now = datetime.now()
        replacements = {
            "{{timestamp}}": now.strftime("%Y-%m-%d %H:%M:%S"),
            "{{date}}": now.strftime("%Y-%m-%d"),
            "{{time}}": now.strftime("%H:%M:%S"),
            "{{datetime}}": now.isoformat(),
        }

        # System placeholders
        window_title = self.system_manager.get_active_window_title()
        if window_title:
            replacements["{{window_title}}"] = window_title

        # Apply replacements
        for placeholder, value in replacements.items():
            prompt = prompt.replace(placeholder, value)

        return prompt

    def should_use_completion(self, prompt_template: str) -> bool:
        """Determine if prompt requires completion or direct transcription.

        Args:
            prompt_template: Prompt template to analyze

        Returns:
            True if completion is needed, False for direct transcription

        """
        # If the prompt only contains {{text}}, just return the transcript
        stripped = prompt_template.strip()
        return stripped != "{{text}}"

    def get_prompt_categories(self, mode: str = "default") -> list[str]:
        """Get available prompt categories.

        Args:
            mode: Prompt mode ('default' or 'dictation')

        Returns:
            List of category names

        """
        prompt_rules = self.config.get_prompts_config(mode)
        categories = set()

        for rule in prompt_rules:
            category = rule.get("category", "General")
            categories.add(category)

        return sorted(list(categories))

    def get_prompts_by_category(
        self,
        category: str,
        mode: str = "default",
    ) -> list[dict[str, Any]]:
        """Get prompts filtered by category.

        Args:
            category: Category name to filter by
            mode: Prompt mode ('default' or 'dictation')

        Returns:
            List of prompt rules in the category

        """
        prompt_rules = self.config.get_prompts_config(mode)
        return [
            rule for rule in prompt_rules if rule.get("category", "General") == category
        ]

    def validate_prompt_template(self, template: str) -> dict[str, Any]:
        """Validate a prompt template.

        Args:
            template: Prompt template to validate

        Returns:
            Validation result with status and messages

        """
        result = {
            "valid": True,
            "warnings": [],
            "errors": [],
        }

        # Check for required {{text}} placeholder
        if "{{text}}" not in template:
            result["warnings"].append("Template does not contain {{text}} placeholder")

        # Check for unbalanced braces
        open_braces = template.count("{{")
        close_braces = template.count("}}")
        if open_braces != close_braces:
            result["valid"] = False
            result["errors"].append("Unbalanced placeholder braces")

        # Check for unknown placeholders
        import re

        placeholders = re.findall(r"\{\{([^}]+)\}\}", template)
        known_placeholders = {
            "text",
            "timestamp",
            "date",
            "time",
            "datetime",
            "window_title",
        }

        for placeholder in placeholders:
            if placeholder not in known_placeholders:
                result["warnings"].append(
                    f"Unknown placeholder: {{{{ {placeholder} }}}}",
                )

        # Check template length
        if len(template) > 5000:
            result["warnings"].append("Template is very long and may cause API issues")

        return result

    def create_custom_prompt(
        self,
        name: str,
        pattern: str,
        template: str,
        category: str = "Custom",
        mode: str = "default",
    ) -> bool:
        """Create a custom prompt rule.

        Args:
            name: Name for the prompt rule
            pattern: Regex pattern to match window titles
            template: Prompt template
            category: Category for organization
            mode: Prompt mode ('default' or 'dictation')

        Returns:
            True if successful, False otherwise

        """
        try:
            # Validate the template
            validation = self.validate_prompt_template(template)
            if not validation["valid"]:
                print(f"Invalid template: {', '.join(validation['errors'])}")
                return False

            # Validate regex pattern
            try:
                re.compile(pattern)
            except re.error as e:
                print(f"Invalid regex pattern: {e}")
                return False

            # Load existing prompts
            prompt_rules = self.config.get_prompts_config(mode)

            # Add new rule
            new_rule = {
                "name": name,
                "match": pattern,
                "prompt": template,
                "category": category,
            }
            prompt_rules.append(new_rule)

            # Save back to config file
            import yaml

            config_file = (
                self.config.config_dir / "dictation.yaml"
                if mode == "dictation"
                else self.config.config_dir / "prompts.yaml"
            )

            with open(config_file, "w") as f:
                yaml.dump(prompt_rules, f, default_flow_style=False)

            return True

        except Exception as e:
            print(f"Error creating custom prompt: {e}")
            return False

    def get_prompt_statistics(self, mode: str = "default") -> dict[str, Any]:
        """Get statistics about prompt usage and configuration.

        Args:
            mode: Prompt mode ('default' or 'dictation')

        Returns:
            Dictionary containing prompt statistics

        """
        prompt_rules = self.config.get_prompts_config(mode)

        stats = {
            "total_prompts": len(prompt_rules),
            "categories": self.get_prompt_categories(mode),
            "category_count": len(self.get_prompt_categories(mode)),
            "has_default": any(rule.get("match", "") == "" for rule in prompt_rules),
            "pattern_types": {
                "empty": len([r for r in prompt_rules if not r.get("match", "")]),
                "regex": len([r for r in prompt_rules if r.get("match", "")]),
            },
        }

        return stats

    def test_prompt_matching(
        self,
        window_title: str,
        mode: str = "default",
    ) -> dict[str, Any]:
        """Test which prompt would be selected for a given window title.

        Args:
            window_title: Window title to test
            mode: Prompt mode ('default' or 'dictation')

        Returns:
            Information about the matched prompt

        """
        prompt_rules = self.config.get_prompts_config(mode)

        for rule in prompt_rules:
            pattern = rule.get("match", "")
            if pattern and re.search(pattern, window_title, re.IGNORECASE):
                return {
                    "matched": True,
                    "rule": rule,
                    "pattern": pattern,
                    "prompt": rule.get("prompt", "{{text}}"),
                    "category": rule.get("category", "General"),
                }

        return {
            "matched": False,
            "rule": None,
            "pattern": "",
            "prompt": "{{text}}",
            "category": "Default",
        }
