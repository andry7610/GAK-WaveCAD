import yaml
from pathlib import Path
from typing import Any, Dict
import re


class ConfigLoader:
    @staticmethod
    def _coerce_types(obj: Any) -> Any:
        """Рекурсивно приводит строковые числа к float/int.
        PyYAML 1.1 парсит 2.0e9 как строку — это фикс."""
        number_re = re.compile(r'^[+-]?\d+\.?\d*([eE][+-]?\d+)?$')
        if isinstance(obj, dict):
            return {k: ConfigLoader._coerce_types(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [ConfigLoader._coerce_types(v) for v in obj]
        if isinstance(obj, str):
            stripped = obj.strip()
            if number_re.match(stripped):
                try:
                    if '.' in stripped or 'e' in stripped or 'E' in stripped:
                        return float(stripped)
                    return int(stripped)
                except ValueError:
                    return obj
        return obj

    @staticmethod
    def load(config_path: str) -> Dict[str, Any]:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not config:
            raise ValueError("Config file is empty")

        config = ConfigLoader._coerce_types(config)

        return config
