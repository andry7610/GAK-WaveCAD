# GAK-WaveCAD

Модульная система моделирования физических процессов в сферической оболочке.

## Архитектура

GAK-WaveCAD/ ├── core/ │ ├── base_module.py # Базовый класс модуля │ ├── config_loader.py # Загрузка YAML-конфигурации │ ├── module_registry.py # Реестр модулей (декоратор @register) │ └── logger.py # Логирование ├── physical_modules/ │ ├── osmosis_monitor.py # Осмотическое давление │ ├── thermal_monitor.py # Тепловые напряжения │ ├── acoustic_monitor.py # Акустические моды │ └── born_collapse_monitor.py # Устойчивость оболочки ├── configs/ │ └── config.yaml # Конфигурация всех модулей ├── tests/ │ ├── run_osmosis_test.py │ ├── run_thermal_test.py │ ├── run_acoustic_test.py │ ├── run_born_collapse_test.py │ └── test_integration.py ├── run.py # Главная точка запуска ├── run_osmosis_test.py # Корневые копии тестов для CI ├── run_thermal_test.py ├── run_acoustic_test.py └── run_born_collapse_test.py

text

## Цепочка модулей

Осмос → Термалка → Акустика → Коллапс │ │ │ │ │ │ │ └─ фаза: STABLE / WARNING / COLLAPSE / INFLATION │ │ └─ сдвиг резонансных частот от суммарного напряжения │ └─ тепловое напряжение σ_thermal └─ осмотическое напряжение σ_osmotic

text

Напряжения от осмоса и термалки суммируются и передаются в акустический модуль.
Осмотическое давление передаётся в модуль коллапса как внешнее.

## Модули

### Osmosis Monitor
Осмотическое давление по формуле Вант-Гоффа: `π = iCRT`.
Напряжение в оболочке: `σ = πR / (2h)`.

### Thermal Monitor
Стационарный тепловой поток через сферическую стенку.
Тепловое напряжение: `σ = EαΔT / (1 - ν)`.

### Acoustic Monitor
Собственные частоты сферической оболочки:
- breathing (l=0) — радиальная мода
- flexural (l=1,2,3) — изгибные моды

Сдвиг частот под нагрузкой: `f = f₀√(1 + σ/E)`.

### Born Collapse Monitor
Критическое давление коллапса (Зоэлли): `P_cr = 2E/√(3(1-ν²)) × (h/R)²`.
Фазы: STABLE, WARNING, COLLAPSE, INFLATION.

## Запуск

```bash
# Установка зависимостей
pip install numpy scipy pyyaml

# Полная цепочка
python run.py

# Отдельные тесты
python run_osmosis_test.py
python run_thermal_test.py
python run_acoustic_test.py
python run_born_collapse_test.py

# Интеграционный тест
python tests/test_integration.py
Конфигурация
Все параметры задаются в configs/config.yaml. Каждый модуль читает свой раздел.

CI
GitHub Actions запускает все *test*.py файлы на Python 3.11 с numpy и scipy.
