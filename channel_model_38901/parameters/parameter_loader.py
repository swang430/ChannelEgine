import json
from typing import Dict, Any

class ParameterTables:
    """A data class to hold all parameter tables loaded from the standard."""
    def __init__(self, file_path: str):
        self.param_file_path = file_path
        self.data: Dict[str, Any] = {}

    def get_scenario_params(self, scenario_name: str) -> Dict[str, Any]:
        return self.data.get("scenarios", {}).get(scenario_name, {})

class ParameterLoader:
    """
    Loads and parses the 3GPP TR 38.901 parameters from a JSON file.
    """
    @staticmethod
    def load_parameters(file_path: str) -> ParameterTables:
        """
        Parses the JSON parameter file and populates a ParameterTables object.

        Args:
            file_path: The absolute path to the JSON parameter file.

        Returns:
            A ParameterTables object filled with the standard's data.
        """
        tables = ParameterTables(file_path)
        with open(file_path, 'r') as f:
            tables.data = json.load(f)
        # Basic validation
        if "standard_version" not in tables.data or "scenarios" not in tables.data:
            raise ValueError("Invalid parameter file format")
        return tables