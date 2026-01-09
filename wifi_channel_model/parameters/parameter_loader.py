import json
from typing import Dict, Any


class WiFiParameterTables:
    """Holds WiFi model parameters loaded from JSON."""
    def __init__(self, file_path: str):
        self.param_file_path = file_path
        self.data: Dict[str, Any] = {}
        self._normalized_keys: Dict[str, str] = {}

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        if not model_name:
            return ""
        normalized = "".join(ch for ch in model_name.upper() if ch.isalnum())
        if normalized.startswith("MODEL"):
            normalized = normalized[5:]
        if normalized.startswith("TGAX"):
            normalized = normalized[4:]
        return normalized

    def build_index(self) -> None:
        self._normalized_keys = {}
        for key in self.data.get("models", {}).keys():
            normalized = self._normalize_model_name(key)
            if normalized:
                self._normalized_keys[normalized] = key
            compact = "".join(ch for ch in key.upper() if ch.isalnum())
            if compact:
                self._normalized_keys[compact] = key

    def get_model(self, model_name: str) -> Dict[str, Any]:
        models = self.data.get("models", {})
        if model_name in models:
            return models.get(model_name, {})
        normalized = self._normalize_model_name(model_name)
        canonical = self._normalized_keys.get(normalized)
        if canonical:
            return models.get(canonical, {})
        compact = "".join(ch for ch in model_name.upper() if ch.isalnum())
        canonical = self._normalized_keys.get(compact)
        return models.get(canonical, {})


class WiFiParameterLoader:
    """Loads WiFi channel model parameters from a JSON file."""
    @staticmethod
    def load_parameters(file_path: str) -> WiFiParameterTables:
        tables = WiFiParameterTables(file_path)
        with open(file_path, "r") as f:
            tables.data = json.load(f)
        if "models" not in tables.data:
            raise ValueError("Invalid WiFi parameter file format")
        tables.build_index()
        return tables
