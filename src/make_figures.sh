#!/usr/bin/env bash
#
# Regenerate every real-data / intro / rate figure used in the paper.
#
#   astro, chemi, intro_figure : real-data figures (share utils.plotting.style)
#   plot_rates                 : simulation rate plots (own style, untouched)
#
# Each real-data script caches its expensive computation under src/fig/cache/,
# so re-running after a styling tweak is fast. To force a full recompute, delete
# the relevant pickle (or set force_recompute=True in the script).
#
# MPLBACKEND=Agg keeps matplotlib non-interactive so plot_rates' plt.show()
# does not block. All figures are written to src/fig/.
set -euo pipefail
cd "$(dirname "$0")"            # -> src/
export MPLBACKEND=Agg

echo "[1/4] astro";        python3 real_data/astro.py
echo "[2/4] chemi";        python3 real_data/chemi.py
echo "[3/4] intro_figure"; python3 intro_figure.py
echo "[4/4] plot_rates";   ( cd simulations && python3 plot_rates.py )

echo "Done. Figures written to src/fig/."