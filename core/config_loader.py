import yaml
from pathlib import Path
from typing import Any, Dict

class ConfigLoader:
    @staticmethod
    def load(config_path: str) -> Dict[str, Any]:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not config:
            raise ValueError("Config file is empty")

        # Базовая валидация — можно расширять под свои нужды
        required_keys = ["modules"]
        for key in required_keys:
            if key not in config:
                raise KeyError(f"Missing required key in config: {key}")

        return config
