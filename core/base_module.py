from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseModule(ABC):
    """
    Базовый класс для всех модулей системы.
    Гарантирует наличие методов init(), run() и get_results().
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._initialized = False
        self._results: Dict[str, Any] = {}

    @abstractmethod
    def init(self) -> bool:
        """
        Инициализация модуля.
        Возвращает True, если инициализация прошла успешно, иначе False.
        Здесь можно загружать зависимости, проверять конфиг, готовить данные.
        """
        pass

    @abstractmethod
    def run(self) -> bool:
        """
        Основной метод выполнения логики модуля.
        Возвращает True, если выполнение прошло успешно, иначе False.
        """
        pass

    @abstractmethod
    def get_results(self) -> Dict[str, Any]:
        """
        Возвращает словарь с результатами работы модуля.
        Структура словаря зависит от конкретного модуля.
        """
        pass

    def is_initialized(self) -> bool:
        """Проверяет, был ли модуль инициализирован."""
        return self._initialized

    def _set_initialized(self, status: bool):
        """Внутренний метод для установки флага инициализации."""
        self._initialized = status
