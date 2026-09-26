from typing import Dict, Type, List, Any
from core.base_module import BaseModule

class ModuleRegistry:
    _registry: Dict[str, Type[BaseModule]] = {}

    @classmethod
    def register(cls, name: str):
        """Декоратор для регистрации модулей в реестре."""
        def decorator(module_class: Type[BaseModule]):
            if not issubclass(module_class, BaseModule):
                raise TypeError(f"Class {module_class.__name__} must inherit from BaseModule")
            cls._registry[name] = module_class
            return module_class
        return decorator

    @classmethod
    def get_module(cls, name: str) -> Type[BaseModule]:
        """Получить класс модуля по имени."""
        if name not in cls._registry:
            raise KeyError(f"Module '{name}' not found in registry. Available: {list(cls._registry.keys())}")
        return cls._registry[name]

    @classmethod
    def list_available(cls) -> List[str]:
        """Список всех зарегистрированных модулей."""
        return list(cls._registry.keys())

    @classmethod
    def create_and_init(cls, name: str, config: Dict[str, Any]) -> BaseModule:
        """Создать экземпляр модуля и вызвать init(). Возвращает готовый к run() модуль."""
        module_class = cls.get_module(name)
        module = module_class(config=config.get("modules", {}).get(name, {}))
        if not module.init():
            raise RuntimeError(f"Initialization failed for module '{name}'")
        return module
