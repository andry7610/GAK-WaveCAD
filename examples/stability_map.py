#!/usr/bin/env python3
"""
GAK-WaveCAD — Example: Stability Map
Карта стабильности системы в координатах температура × давление.

Запуск:
    python examples/stability_map.py
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
from physical_modules.born_collapse_monitor import BornCollapseMonitor
from physical_modules.magnon_monitor import MagnonMonitor
from physical_modules.em_resonance_monitor import EMResonanceMonitor
from physical_modules.coupling_monitor import CouplingMonitor


def main():
    print("=" * 64)
    print("  GAK-WaveCAD — Stability Map")
    print("  Температура × Концентрация → Stability Index")
    print("=" * 64)

    # Сетки параметров
    temps = np.linspace(250, 450, 30)       # К
    concentrations = np.linspace(0.1, 3.0, 30)  # моль/л

    T_grid, C_grid = np.meshgrid(temps, concentrations)
    stability_grid = np.zeros_like(T_grid)

    total_cells = len(temps) * len(concentrations)
    done = 0

    for i in range(len(concentrations)):
        for j in range(len(temps)):
            C = C_grid[i, j]
            T = T_grid[i, j]

            try:
                # Осмос
                osmosis = OsmosisMonitor({'C_in': C, 'C_out': 0.01})
                osmosis.init()
                osmosis.run()
                osm_stress = osmosis.get_stress_Pa()
                osm_pressure = osmosis.results['pi_Pa']

                # Термалка
                thermal = ThermalMonitor({'T_inner': T, 'T_outer': 300})
                thermal.init()
                thermal.run()
                th_stress = thermal.get_thermal_stress_Pa()

                total = osm_stress + th_stress

                # Акустика
                acoustic = AcousticMonitor()
                acoustic.set_internal_stress(total)
                acoustic.init()
                acoustic.run()

                # Коллапс
                collapse = BornCollapseMonitor()
                collapse.set_external_pressure(osm_pressure)
                collapse.init()
                collapse.run()

                # Магноны
                magnon = MagnonMonitor()
                magnon.set_mechanical_stress(total)
                magnon.init()
                magnon.run()

                # EM
                em = EMResonanceMonitor()
                em.set_mechanical_stress(total)
                em.init()
                em.run()

                # Кросс-связи
                coupling = CouplingMonitor()
                coupling.init()
                coupling.set_results(
                    osmosis=osmosis.get_results(),
                    thermal=thermal.get_results(),
                    acoustic=acoustic.get_results(),
                    collapse=collapse.get_results(),
                    magnon=magnon.get_results(),
                    em=em.get_results(),
                )
                coupling.run()
                stability_grid[i, j] = coupling.get_results()['stability_index']
            except Exception:
                stability_grid[i, j] = 0.0

            done += 1
            if done % 100 == 0:
                print(f"  Прогресс: {done}/{total_cells}")

    # График
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(stability_grid, origin='lower', aspect='auto', cmap='RdYlGn',
                  extent=[temps[0], temps[-1], concentrations[0], concentrations[-1]],
                  vmin=0, vmax=1)

    ax.set_xlabel('Температура, К', fontsize=12)
    ax.set_ylabel('Концентрация NaCl, моль/л', fontsize=12)
    ax.set_title('Карта стабильности системы', fontsize=13)

    # Изолинии
    cs = ax.contour(T_grid, C_grid, stability_grid,
                    levels=[0.4, 0.7], colors=['black', 'black'],
                    linewidths=1.5, linestyles=['dashed', 'solid'])
    ax.clabel(cs, fmt={0.4: 'CRITICAL', 0.7: 'HEALTHY'}, fontsize=9)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Stability Index', fontsize=11)

    output = os.path.join(os.path.dirname(__file__), "stability_map.png")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()

    print(f"\n  График сохранён: {output}")


if __name__ == "__main__":
    main()
