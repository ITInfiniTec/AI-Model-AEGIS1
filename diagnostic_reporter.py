# diagnostic_reporter.py

from typing import List, Dict, Any

class DiagnosticReporter:
    """A utility to collect and report on internal warnings and corrections."""
    def __init__(self):
        self.warnings: List[str] = []
        self.corrections: List[str] = []

    def add_warning(self, source: str, message: str):
        """Adds a warning to the report."""
        self.warnings.append(f"[{source}] {message}")

    def add_correction(self, source: str, action: str):
        """Adds a record of a corrective action to the report."""
        self.corrections.append(f"[{source}] {action}")

    def generate_report(self) -> Dict[str, Any]:
        """Generates the final debug report dictionary."""
        return {
            "warnings": self.warnings,
            "corrections_made": self.corrections,
        }