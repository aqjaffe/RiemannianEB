"""Shared visual style for the real-data / intro figures.

Applies to astro, chemi and intro_figure ONLY — the simulation rate plots
(plot_rates / plot_mcratesims_o) are intentionally left untouched.

Centralising these constants keeps every comparable panel consistent in marker
alpha, marker size, title size, tick size and inter-panel spacing. Tweak a value
here and re-run; because each figure script caches its (expensive) plot data,
restyling is effectively instant.
"""

SCATTER_ALPHA = 0.12    # marker transparency for scatter panels 
SCATTER_SIZE  = 10      # marker size for scatter panels
TITLE_SIZE    = 21      # subplot title fontsize
TICK_SIZE     = 15      # tick-label fontsize
LABEL_SIZE    = 18      # loss / axis-label fontsize

PANEL_SIZE    = 4.0     # nominal inches per square panel
WSPACE        = 0.12    # horizontal gap between panels (fraction of panel width)
HSPACE        = 0.28    # vertical gap between panels
TITLE_Y       = 1.08    # title vertical placement for mollweide panels