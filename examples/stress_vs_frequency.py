#!/usr/bin/env python3
"""
GAK-WaveCAD — Example: Stress vs Frequency
Как изменение концентрации NaCl влияет на акустические частоты.

Запуск:
    python examples/stress_vs_frequency.py
Требует: numpy, matplotlib
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from physical_modules.osmosis_monitor import OsmosisMonitor
from physical_modules.thermal_monitor import ThermalMonitor
from physical_modules.acoustic_monitor import AcousticMonitor


def main():
    print("=" * 64)
    print("  GAK-WaveCAD — Stress vs Frequency")
    print("  Концентрация NaCl → акустические частоты")
    print("=" * 64)

    # Диапазон концентраций, моль/л
    concentrations = np.linspace(0.1, 2.0, 20)

    # Базовые модули (термалка — фиксированная)
    thermal = ThermalMonitor()
    thermal.init()
    thermal.run()
    th_stress = thermal.get_thermal_stress_Pa()

    # Для каждой моды — список частот
    mode_names = None
    freqs_by_mode = {}

    for i, C in enumerate(concentrations):
        # Осмос с меняющейся концентрацией
        osmosis = OsmosisMonitor({'C_in': C, 'C_out': 0.01})
        osmosis.init()
        osmosis.run()
        osm_stress = osmosis.get_stress_Pa()
        total = osm_stress + th_stress

        # Акустика
        acoustic = AcousticMonitor()
        acoustic.set_internal_stress(total)
        acoustic.init()
        acoustic.run()
        ac = acoustic.get_results()

        if mode_names is None:
            mode_names = list(ac.keys())
            freqs_by_mode = {m: [] for m in mode_names}

        for m in mode_names:
            freqs_by_mode[m].append(ac[m]['f_measured'])

        if (i + 1) % 5 == 0:
            print(f"  C={C:.2f} М → σ={total/1e6:.2f} МПа")

    # График
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#4a90d9', '#2ecc71', '#e67e22', '#e74c3c']
    for i, m in enumerate(mode_names):
        ax.plot(concentrations, freqs_by_mode[m], 'o-', color=colors[i % 4],
                label=m.replace('_', ' '), linewidth=2, markersize=4)

    ax.set_xlabel('Концентрация NaCl, моль/л', fontsize=12)
    ax.set_ylabel('Частота, Гц', fontsize=12)
    ax.set_title('Сдвиг акустических частот от концентрации', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    output = os.path.join(os.path.dirname(__file__), "stress_vs_frequency.png")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()

    print(f"\n  График сохранён: {output}")


if __name__ == "__main__":
    main()
