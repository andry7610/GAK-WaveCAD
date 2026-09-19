#!/usr/bin/env python3
"""
GAK-WaveCAD — Demo Script
Прогоняет всю цепочку из 7 модулей и строит графики.

Запуск:
    python demo.py
Требует: numpy, matplotlib
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from core.config_loader import ConfigLoader
from core.logger import get_logger
from physical_modules.osmosis_monitor import OsmosisMonitor
from physical_modules.thermal_monitor import ThermalMonitor
from physical_modules.acoustic_monitor import AcousticMonitor
from physical_modules.born_collapse_monitor import BornCollapseMonitor
from physical_modules.magnon_monitor import MagnonMonitor
from physical_modules.em_resonance_monitor import EMResonanceMonitor
from physical_modules.coupling_monitor import CouplingMonitor


def main():
    logger = get_logger("demo")

    config_path = os.path.join(os.path.dirname(__file__), "configs", "config.yaml")
    config = ConfigLoader.load(config_path)

    # --- 1. Осмос ---
    osmosis = OsmosisMonitor(config.get("osmosis_monitor", {}))
    osmosis.init()
    osmosis.run()
    osm_stress = osmosis.get_stress_Pa()
    osm_pressure = osmosis.results['pi_Pa']

    # --- 2. Термалка ---
    thermal = ThermalMonitor(config.get("thermal_monitor", {}))
    thermal.init()
    thermal.run()
    th_stress = thermal.get_thermal_stress_Pa()

    total_stress = osm_stress + th_stress
    logger.info(f"σ_osm={osm_stress:.1f}, σ_th={th_stress:.1f}, Σ={total_stress:.1f} Па")

    # --- 3. Акустика ---
    acoustic = AcousticMonitor(config.get("acoustic_monitor", {}))
    acoustic.set_internal_stress(total_stress)
    acoustic.init()
    acoustic.run()
    ac_results = acoustic.get_results()

    # --- 4. Коллапс ---
    collapse = BornCollapseMonitor(config.get("born_collapse_monitor", {}))
    collapse.set_external_pressure(osm_pressure)
    collapse.init()
    collapse.run()

    # --- 5. Магноны ---
    magnon = MagnonMonitor(config.get("magnon_monitor", {}))
    magnon.set_mechanical_stress(total_stress)
    magnon.init()
    magnon.run()
    mag_results = magnon.get_results()

    # --- 6. EM ---
    em = EMResonanceMonitor(config.get("em_resonance_monitor", {}))
    em.set_mechanical_stress(total_stress)
    em.init()
    em.run()
    em_results = em.get_results()

    # --- 7. Кросс-связи ---
    coupling = CouplingMonitor()
    coupling.init()
    coupling.set_results(
        osmosis=osmosis.get_results(),
        thermal=thermal.get_results(),
        acoustic=ac_results,
        collapse=collapse.get_results(),
        magnon=mag_results,
        em=em_results,
    )
    coupling.run()
    coupl = coupling.get_results()

    # ============================================================
    # Графики
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        'GAK-WaveCAD — Demo Simulation\n'
        'осмос → термалка → акустика → коллапс → магноны → EM → кросс-связи',
        fontsize=13, fontweight='bold'
    )
    width = 0.35

    # 1. Акустические моды
    ax = axes[0, 0]
    ac_labels = list(ac_results.keys())
    f0 = [r['f_reference'] for r in ac_results.values()]
    f1 = [r['f_measured'] for r in ac_results.values()]
    x = np.arange(len(ac_labels))
    ax.bar(x - width/2, f0, width, label='f₀', color='#4a90d9', alpha=0.8)
    ax.bar(x + width/2, f1, width, label='f (σ)', color='#e74c3c', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace('_', '\n') for l in ac_labels], fontsize=8)
    ax.set_ylabel('Частота, Гц')
    ax.set_title('Акустические моды')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # 2. Магнонный спектр
    ax = axes[0, 1]
    mag_labels = list(mag_results.keys())
    f_mag = [r['f'] / 1e9 for r in mag_results.values()]
    f_mag_ns = [(r['f'] - r['df_stress']) / 1e9 for r in mag_results.values()]
    x = np.arange(len(mag_labels))
    ax.bar(x - width/2, f_mag_ns, width, label='f₀ (без σ)', color='#2ecc71', alpha=0.8)
    ax.bar(x + width/2, f_mag, width, label='f (с σ)', color='#e67e22', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace('_', '\n') for l in mag_labels], fontsize=8)
    ax.set_ylabel('Частота, ГГц')
    ax.set_title('Магнонный спектр (YIG)')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # 3. EM-моды
    ax = axes[0, 2]
    em_labels = list(em_results.keys())
    f_em = [r['f_shifted'] / 1e9 for r in em_results.values()]
    f_em0 = [r['f_base'] / 1e9 for r in em_results.values()]
    x = np.arange(len(em_labels))
    ax.bar(x - width/2, f_em0, width, label='f₀ (без σ)', color='#9b59b6', alpha=0.8)
    ax.bar(x + width/2, f_em, width, label='f (с σ)', color='#e74c3c', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace('_', '\n') for l in em_labels], fontsize=8)
    ax.set_ylabel('Частота, ГГц')
    ax.set_title('EM-резонанс (TM/TE)')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # 4. Относительные сдвиги (ppm)
    ax = axes[1, 0]
    ac_rel = [r['delta_f'] / r['f_reference'] * 1e6 for r in ac_results.values()]
    mag_rel = [r['df_stress'] / (r['f'] - r['df_stress']) * 1e6 for r in mag_results.values()]
    em_rel = [r['df_stress'] / r['f_base'] * 1e6 for r in em_results.values()]
    n = max(len(ac_rel), len(mag_rel), len(em_rel))
    ax.plot(range(len(ac_rel)), ac_rel, 'o-', label='Акустика', color='#4a90d9', linewidth=2)
    ax.plot(range(len(mag_rel)), mag_rel, 's-', label='Магноны', color='#e67e22', linewidth=2)
    ax.plot(range(len(em_rel)), em_rel, '^-', label='EM', color='#9b59b6', linewidth=2)
    ax.set_xlabel('Номер моды')
    ax.set_ylabel('Относительный сдвиг, ppm')
    ax.set_title(f'Сдвиг частот (σ = {total_stress/1e6:.1f} МПа)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 5. Кросс-связи
    ax = axes[1, 1]
    c_labels = ['mag-elastic\n(ак↔маг)', 'piezo\n(ак↔EM)', 'mag-electric\n(маг↔EM)', 'thermal-osm\n(т↔осм)']
    c_vals = [
        min(coupl['k_magnetoelastic'], 1.0),
        min(coupl['k_piezoelectric'], 1.0),
        min(coupl['k_magnetoelectric'], 1.0),
        min(coupl['k_thermal_osmosis'], 1.0),
    ]
    c_colors = ['#e74c3c', '#9b59b6', '#2ecc71', '#4a90d9']
    ax.barh(range(4), c_vals, color=c_colors, alpha=0.8, height=0.5)
    ax.set_yticks(range(4))
    ax.set_yticklabels(c_labels, fontsize=9)
    ax.set_xlabel('Коэффициент связи')
    ax.set_title('Кросс-связи')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, 1.05)

    # 6. Stability gauge
    ax = axes[1, 2]
    ax.set_aspect('equal')
    ax.axis('off')
    r_out, r_in = 1.0, 0.7
    for ts, te, col in [(np.pi, np.pi*0.6, '#e74c3c'),
                         (np.pi*0.6, np.pi*0.3, '#f39c12'),
                         (np.pi*0.3, 0, '#2ecc71')]:
        arc = np.linspace(ts, te, 50)
        xo = r_out * np.cos(arc)
        yo = r_out * np.sin(arc)
        xi = r_in * np.cos(arc[::-1])
        yi = r_in * np.sin(arc[::-1])
        ax.fill(np.concatenate([xo, xi]), np.concatenate([yo, yi]), color=col, alpha=0.6)

    stab = coupl['stability_index']
    ang = np.pi * (1 - stab)
    ax.annotate('', xy=(np.cos(ang)*0.85, np.sin(ang)*0.85), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='black', lw=3))
    status = coupl['system_status']
    sc = {'HEALTHY': '#2ecc71', 'DEGRADED': '#f39c12', 'CRITICAL': '#e74c3c'}[status]
    ax.text(0, -0.15, f'Stability = {stab:.3f}', ha='center', fontsize=12, fontweight='bold')
    ax.text(0, -0.28, status, ha='center', fontsize=14, fontweight='bold', color=sc)
    ax.text(-0.85, 0.35, 'CRITICAL', fontsize=8, color='#e74c3c', ha='center')
    ax.text(0, 0.55, 'DEGRADED', fontsize=8, color='#f39c12', ha='center')
    ax.text(0.85, 0.35, 'HEALTHY', fontsize=8, color='#2ecc71', ha='center')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.35, 1.1)
    ax.set_title('Индекс стабильности')

    plt.tight_layout()
    output = os.path.join(os.path.dirname(__file__), "demo_output.png")
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"График сохранён: {output}")
    print(f"\n  График сохранён: {output}")
    print(f"  Stability = {stab:.3f}, Status = {status}")


if __name__ == "__main__":
    main()
