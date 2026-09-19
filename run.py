"""
GAK-WaveCAD — точка входа.

Загружает конфиг, инициализирует модули через реестр и запускает их.
"""

from core.config_loader import ConfigLoader
from core.module_registry import ModuleRegistry

# Импортируем модуль, чтобы сработал декоратор @ModuleRegistry.register
from physical_modules.acoustic_monitor import AcousticMonitor


def main():
    # Загружаем конфиг
    config = ConfigLoader.load("configs/config.yaml")
    print("Конфиг загружен.")
    print(f"Доступные модули: {ModuleRegistry.list_available()}")

    # Создаём и инициализируем модуль
    module = ModuleRegistry.create_and_init("acoustic_monitor", config)
    print("Модуль acoustic_monitor инициализирован.")

    # Запускаем
    if module.run():
        print("Анализ завершён.\n")
        results = module.get_results()
        for key, r in results.items():
            print(f"  {key:12s} | {r['status']:16s} | "
                  f"f = {r['f_measured']:.6f} Гц | "
                  f"Δf = {r['delta_f']:.6f} Гц | "
                  f"σ = {r['sigma_MPa']:.4f} МПа | "
                  f"{r['n_peaks']} пик(ов)")
    else:
        print("Ошибка при выполнении модуля.")


if __name__ == "__main__":
    main()
