# GAK-WaveCAD

Модульная система моделирования физических процессов в сферической оболочке.

## Архитектура

    GAK-WaveCAD/
    ├── core/
    │   ├── base_module.py           # Базовый класс модуля
    │   ├── config_loader.py         # Загрузка YAML-конфигурации
    │   ├── module_registry.py       # Реестр модулей (декоратор @register)
    │   └── logger.py                # Логирование
    ├── physical_modules/
    │   ├── osmosis_monitor.py       # Осмотическое давление
    │   ├── thermal_monitor.py       # Тепловые напряжения
    │   ├── acoustic_monitor.py      # Акустические моды
    │   ├── born_collapse_monitor.py # Устойчивость оболочки
    │   ├── magnon_monitor.py        # Спиновые моды (YIG)
    │   ├── em_resonance_monitor.py  # Электромагнитные моды
    │   └── coupling_monitor.py      # Кросс-связи между модулями
    ├── configs/
    │   └── config.yaml              # Конфигурация всех модулей
    ├── tests/
    │   ├── run_osmosis_test.py
    │   ├── run_thermal_test.py
    │   ├── run_acoustic_test.py
    │   ├── run_born_collapse_test.py
    │   ├── run_magnon_test.py
    │   ├── run_em_resonance_test.py
    │   ├── run_coupling_test.py
    │   └── test_integration.py
    ├── demo.py                      # Демо-скрипт с графиками
    ├── run.py                       # Главная точка запуска
    ├── run_osmosis_test.py          # Корневые копии тестов для CI
    ├── run_thermal_test.py
    ├── run_acoustic_test.py
    ├── run_born_collapse_test.py
    ├── run_magnon_test.py
    ├── run_em_resonance_test.py
    └── run_coupling_test.py

## Цепочка модулей

    Осмос → Термалка → Акустика → Коллапс → Магноны → EM → Кросс-связи
      │         │          │          │          │        │        │
      │         │          │          │          │        │        └─ stability index, system status
      │         │          │          │          │        └─ TM/TE-моды, пьезосдвиг через d_piezo
      │         │          │          │          └─ Kittel + обменные моды, сдвиг через λ_s
      │         │          │          └─ фаза: STABLE / WARNING / COLLAPSE / INFLATION
      │         │          └─ сдвиг резонансных частот от суммарного напряжения
      │         └─ тепловое напряжение σ_thermal
      └─ осмотическое напряжение σ_osmotic

Напряжения от осмоса и термалки суммируются и передаются в акустический, магнонный и EM-модуули.
Осмотическое давление передаётся в модуль коллапса как внешнее.
Coupling Monitor собирает результаты всех шести модулей и считает кросс-связи.

## Модули

### Osmosis Monitor
Осмотическое давление по формуле Вант-Гоффа: π = iCRT. Напряжение в оболочке: σ = πR / (2h).

### Thermal Monitor
Стационарный тепловой поток через сферическую стенку. Тепловое напряжение: σ = EαΔT / (1 - ν).

### Acoustic Monitor
Собственные частоты сферической оболочки:
- breathing (l=0) — радиальная мода
- flexural (l=1,2,3) — изгибные моды

Сдвиг частот под нагрузкой: f = f₀√(1 + σ/E).

### Born Collapse Monitor
Критическое давление коллапса (Зоэлли): P_cr = 2E/√(3(1-ν²)) × (h/R)². Фазы: STABLE, WARNING, COLLAPSE, INFLATION.

### Magnon Monitor
Собственные частоты спиновых мод в YIG-оболочке:
- kittel (l=0) — однородная прецессия (формула Киттеля)
- exchange (l=1,2,3) — обменные моды

Магнитоупругая связь: механическое напряжение сдвигает частоты через магнитострикцию λ_s.
Добротность Q определяется затуханием Гилберта α.

### EM Resonance Monitor
Электромагнитные моды сферического резонатора:
- TM-моды (корни производной сферической функции Бесселя)
- TE-моды (корни самой функции)

Пьезоэлектрическая связь: механическое напряжение генерирует электрическое поле, сдвигающее EM-частоты через коэффициент d_piezo.

### Coupling Monitor
Метамодуль — считает кросс-связи между всеми модулями:
- k_magnetoelastic — акустика ↔ магноны
- k_piezoelectric — акустика ↔ EM
- k_magnetoelectric — магноны ↔ EM
- k_thermal_osmosis — термалка ↔ осмос
- geom_factor — влияние коллапса на геометрию

Выдаёт общий индекс стабильности (0..1) и системный статус: HEALTHY, DEGRADED, CRITICAL.

## Запуск

    # Установка зависимостей
    pip install numpy scipy pyyaml matplotlib

    # Полная цепочка
    python run.py

    # Демо с графиками
    python demo.py

    # Отдельные тесты
    python run_osmosis_test.py
    python run_thermal_test.py
    python run_acoustic_test.py
    python run_born_collapse_test.py
    python run_magnon_test.py
    python run_em_resonance_test.py
    python run_coupling_test.py

    # Интеграционный тест
    python tests/test_integration.py

## Конфигурация

Все параметры задаются в `configs/config.yaml`. Каждый модуль читает свой раздел.

## CI

GitHub Actions запускает все `*test*.py` файлы на Python 3.11 с numpy и scipy.

