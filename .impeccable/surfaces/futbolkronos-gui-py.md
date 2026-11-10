---
version: 1
slug: "futbolkronos-gui-py"
primary_target: "futbolkronos_gui.py"
related_targets: []
---

# Surface: main simulator window (futbolkronos_gui.py)

Mode: Operate. Single user (the author) evaluating the model. Task: pick a match, simulate, read the distribution and compare it with the market and with the real result. Constraints: Tkinter + ttk + matplotlib, no new dependencies, `simulador.py` untouched, UI in Spanish (Spain).

User choice: category standard, played straight. Craft bar: The Analyst (Opta), the clarity of its prediction graphics. Without copying its branding.

## Direction contract

THESIS: a serious match-prediction panel that reads in one glance; it refuses the generic dashboard of five identical white KPI cards with an indigo accent and scientific-report charts.

OWN-WORLD: neutral dark ground (#0F1115) with a second, slightly lighter layer for the sidebar; one family (Segoe UI Variable); colour only for data: local blue, draw grey, away orange; the primary button is light-on-dark, never a data colour; hit green / miss red only on the real result.

STORY: the user picks a match, simulates, and understands who is favourite, how sure the model is, and whether it got it right, without opening the tabs.

FIRST VIEWPORT: sidebar of controls on the left with hairline dividers; the main area opens with the match header (local and away names large, date and competition), and beneath it the horizontal 1·X·2 bar split proportionally with labelled percentages, the model band under each segment, and a data row (most likely score, expected goals, real result). Below that, tabs with the charts.

FORM: category standard (canon), chosen by the user; seed:72f4401c6809faca.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
