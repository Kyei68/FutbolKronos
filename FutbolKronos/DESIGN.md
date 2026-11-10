---
name: FutbolKronos
description: Desktop match-simulation lab; dark neutral ground where only the data carries colour.
colors:
  ground: "#0F1115"
  sidebar-layer: "#151920"
  surface: "#171B22"
  zebra-row: "#1C212A"
  field: "#1E232C"
  field-hover: "#262C37"
  hairline: "#262C36"
  chart-grid: "#222832"
  selection: "#243650"
  ink: "#E7EAEF"
  ink-muted: "#9BA3B0"
  ink-faint: "#7D8695"
  ink-on-light: "#0B1220"
  home-blue: "#4C9BFF"
  draw-grey: "#8B94A3"
  away-orange: "#FF8A4C"
  hit-green: "#43C07E"
  miss-red: "#F26461"
  real-gold: "#F2C14E"
typography:
  display:
    fontFamily: "Segoe UI Variable Display, Segoe UI, sans-serif"
    fontSize: "22pt"
    fontWeight: 700
  headline:
    fontFamily: "Segoe UI Variable Display, Segoe UI, sans-serif"
    fontSize: "17pt"
    fontWeight: 700
  title:
    fontFamily: "Segoe UI Variable Display, Segoe UI, sans-serif"
    fontSize: "13pt"
    fontWeight: 700
  body:
    fontFamily: "Segoe UI Variable Text, Segoe UI, sans-serif"
    fontSize: "10pt"
    fontWeight: 400
  label:
    fontFamily: "Segoe UI Variable Text, Segoe UI, sans-serif"
    fontSize: "9pt"
    fontWeight: 400
  mono:
    fontFamily: "Cascadia Mono, Consolas, monospace"
    fontSize: "10pt"
    fontWeight: 400
rounded:
  none: "0px"
  pill: "999px"
spacing:
  xs: "3px"
  sm: "8px"
  md: "14px"
  lg: "18px"
  xl: "22px"
  xxl: "26px"
components:
  button-primary:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.ground}"
    typography: "{typography.body}"
    rounded: "{rounded.none}"
    padding: "10px 12px"
  button-primary-hover:
    backgroundColor: "#FFFFFF"
    textColor: "{colors.ground}"
  button-primary-pressed:
    backgroundColor: "#C9CFD8"
    textColor: "{colors.ground}"
  button-primary-disabled:
    backgroundColor: "#2E3440"
    textColor: "{colors.ink-faint}"
  button-ghost:
    backgroundColor: "{colors.field}"
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
    padding: "6px 8px"
  button-ghost-hover:
    backgroundColor: "{colors.field-hover}"
    textColor: "{colors.ink}"
  button-ghost-disabled:
    backgroundColor: "{colors.sidebar-layer}"
    textColor: "{colors.ink-faint}"
  input-field:
    backgroundColor: "{colors.field}"
    textColor: "{colors.ink}"
    typography: "{typography.body}"
    rounded: "{rounded.none}"
    padding: "5px 8px"
  input-field-hover:
    backgroundColor: "{colors.field-hover}"
  tab:
    backgroundColor: "{colors.ground}"
    textColor: "{colors.ink-muted}"
    typography: "{typography.body}"
    padding: "8px 16px"
  tab-selected:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
  table-row:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    typography: "{typography.body}"
    height: "28px"
  table-row-alt:
    backgroundColor: "{colors.zebra-row}"
  table-row-selected:
    backgroundColor: "{colors.selection}"
  checkbox:
    backgroundColor: "{colors.field}"
    size: "18px"
  checkbox-checked:
    backgroundColor: "{colors.home-blue}"
    textColor: "{colors.ink-on-light}"
    size: "18px"
  outcome-bar:
    height: "12px"
    rounded: "{rounded.none}"
---

# Design System: FutbolKronos

## Overview

**Creative North Star: "The Match Analyst's Desk"**

FutbolKronos is a working instrument for one person who opens it over and over to judge a model. The system is the category standard of football prediction graphics played straight, measured against the clarity of The Analyst (Opta) without borrowing its branding: a neutral near-black ground, one native type family, and colour reserved for the data itself. Nothing on screen is decorative; every hue answers a question (who is home, who is away, did the model get it right).

Density is moderate and desktop-native: a fixed 330 px control sidebar on a second, slightly lighter layer, and a main area that opens with a measured, canvas-drawn match header (team names, the proportional 1·X·2 bar with the model's uncertainty band, and a data row), followed by tabs for charts, tables and context. Depth comes from tonal layering and 1 px hairlines, never from shadows. Uncertainty is always shown, not hidden: band brackets over the bar, "modelo 31–35 %" sublabels, and a standing footer note that these are probabilities, not certainties.

The system explicitly rejects the generic dashboard: rows of identical white KPI cards with an indigo accent, and charts that look like a scientific report.

**Key Characteristics:**
- Two-layer dark neutral ground; tonal steps, no shadows.
- Colour is data: home blue, draw grey, away orange, everywhere they appear (bar, charts, Elo figures, convergence lines).
- The primary action is light-on-dark, never a data colour.
- Green/red judge outcomes (real result, recent W/L form); gold marks the real score.
- One native family (Segoe UI Variable) at two optical sizes, plus a mono for tabular form rows.
- Square corners on every control; round only on the slider (track caps and thumb).
- Spanish copy with comma decimals and dot thousands; percentages spaced ("33,5 %").

## Colors

A cool, desaturated slate ground in four close tonal steps, with three saturated data hues and three judgement hues that appear only where data or verdicts live.

### Primary
- **Home Blue** (home-blue): the local team. Its segment of the 1·X·2 bar, its percentage, its chart bars and convergence line, its Elo figure. Doubles as the system's one interaction accent: focus borders on fields and buttons, the slider focus ring, the checked checkbox, the progress bar, and the top of the score-heatmap ramp.

### Secondary
- **Away Orange** (away-orange): the visiting team, in exactly the places home blue marks the local team.
- **Draw Grey** (draw-grey): the draw. A deliberately non-committal neutral; in the header its percentage is set in ink rather than grey so it stays legible.

### Tertiary
- **Hit Green** (hit-green): the model's favourite matched reality ("acertado"), and wins (V) in the recent-form list.
- **Miss Red** (miss-red): the favourite did not match reality ("no era el más probable"), and losses (D) in the recent-form list.
- **Real Gold** (real-gold): the outline box around the real score on the score heatmap, and the low-history warning text in Context.

### Neutral
- **Ground** (ground): main area background, tab strip, unselected tabs.
- **Sidebar Layer** (sidebar-layer): the control sidebar, one step lighter than the ground.
- **Surface** (surface): tab panes, tables, chart figures (and exported PNGs), the selected tab.
- **Zebra Row** (zebra-row): alternate table rows.
- **Field** (field): inputs, combobox, ghost buttons, the empty-state outcome bar, the cold end of the heatmap ramp.
- **Field Hover** (field-hover): field and ghost-button hover.
- **Hairline** (hairline): every 1 px divider and border, the slider track, sidebar/main separator.
- **Chart Grid** (chart-grid): matplotlib gridlines, one step quieter than hairlines.
- **Selection** (selection): selected table rows and combobox list selection; a dark blue so selected text stays ink.
- **Ink** (ink): primary text and the primary button's fill.
- **Ink Muted** (ink-muted): field labels, tick labels, sublabels, band brackets, inactive tabs.
- **Ink Faint** (ink-faint): metadata line, hints, empty placeholders ("—"), disabled text.
- **Ink On Light** (ink-on-light): the checkmark and heatmap values on its brightest cells.

### Named Rules
**The Data Owns Colour Rule.** Saturated hue appears only on data or verdicts. Chrome is ground, hairline, field and ink. If a new element is not a team, an outcome, or the real result, it does not get a hue (focus, the one exception, borrows home blue at 1 to 1.5 px).

**The Light Button Rule.** The primary action is ink on ground (`button-primary`). It is never home blue, away orange or any judgement colour, so it can never be read as a prediction.

**The Fixed Sides Rule.** Home is always blue and left; away is always orange and right; draw is grey and between. Never swap or reuse these hues for anything else.

## Typography

**Display Font:** Segoe UI Variable Display (with Segoe UI)
**Body Font:** Segoe UI Variable Text (with Segoe UI)
**Label/Mono Font:** Cascadia Mono (with Consolas), for tabular form rows only

**Character:** The Windows-native family at its two optical sizes: Display for the few large figures that must read across the room, Text for everything else. It reads as a serious instrument that belongs on the desktop it runs on. Sizes are Tk points; the app sets `tk scaling` to 1.25.

### Hierarchy
- **Display** (700, 22pt): team names in the match header, local left, away right.
- **Headline** (700, 17pt / 15pt): the 1·X·2 percentages under the bar; the favourite gets 17pt, the other two 15pt (offset 2 px lower so baselines settle). Brand wordmark (16pt) and Context title (15pt) sit in this band.
- **Title** (700, 13pt): data-row values (most likely score, expected goals, real result, market).
- **Body** (400, 10pt; 700 for group headings and slider values): fields, tables, tabs, context text. Table headers are 9pt bold in ink-muted.
- **Label** (400, 9pt): field labels, hints, metadata line, data-row captions and sublabels.
- **Mono** (400, 10pt): dates, scores and home/away in the recent-form list, so columns align.
- **Charts** (matplotlib, Segoe UI): 9pt ticks and labels, 10.5pt bold titles left-aligned, 8.5 to 9.5pt bold value labels.

### Named Rules
**The Size Is Confidence Rule.** Only the model's favourite outcome is promoted to the larger headline size. Size encodes the prediction; it is never used for emphasis elsewhere in the header.

**The Measured Text Rule.** Canvas-drawn text is positioned from real font metrics (linespace, measure), never from fixed pixel offsets, so nothing clips at another display scale.

## Layout

A fixed two-column desktop window (opens at 1400×880, minimum 1160×740). Left: a 330 px sidebar (padding 22 px sides, 20 px top, 18 px bottom; wrap width 284 px for hints) separated from the main area by a 1 px hairline. Right: the main area with 26 px left and 22 px right padding, 18 px top, 14 px bottom.

The main area stacks top to bottom: the match header canvas (height computed from its contents), 12 px gap, a notebook of four tabs (Simulación, Marcadores, Goles y mercados, Contexto del partido) with export ghost buttons pinned to the tab strip's right edge, and a single-line disclaimer pinned to the bottom (10 px above it).

The match header is a full-width ladder: metadata line, team names, 16 px, the 12 px-tall 1·X·2 bar (3 px gaps between segments; band brackets 7 px above it), percentages, sublabels, 14 px, a hairline, then the data row split into equal columns separated by vertical hairlines (20 px inset after each divider). In the empty state the same geometry is drawn with placeholders ("—", field-coloured bar) so nothing jumps when a simulation lands.

Sidebar rhythm: groups are separated by a hairline with 18 px above and 14 px below, heading 8 px above its first field; label-to-field 3 px; field-to-next-label 10 px; the date field spans the sidebar width. The primary button has 18 px above it, the 3 px progress bar 8 px below.

Content panes: tables inset 14 px horizontal, 10 px vertical; Context text pane 28 px horizontal, 22 px vertical padding; chart canvas 6 px inset, 2×2 grid.

## Elevation & Depth

Flat. Depth is conveyed by four tonal steps (ground, sidebar layer, surface, field) and 1 px hairlines. The selected tab rises only by taking the surface colour so it merges with its pane. The window title bar follows the active mode (Windows dark or light) so the frame belongs to the same ground. The one shadow in the system is the soft 1 px drop under the theme switch's white knob, which is part of the iOS switch idiom, not a depth layer.

## Light and Dark Modes

Two palettes with identical roles live in `TEMAS` (`oscuro`, the default, and `claro`); `aplicar_tema()` swaps the module constants, so every drawing routine reads the active mode. In light mode the main ground is #F3F4F6 with white sidebar and surfaces, ink #15181D, and the data hues deepen for contrast on white (home #1F6FE5, away #E2621B, hit #16924F, miss #D33A35, real #C88A04). The primary button stays "ink on ground", so it becomes dark on light. The heatmap ramp runs pale grey → light blue → home blue → navy, with white values on its darkest cells.

Switching rebuilds the window with the new palette while keeping the form (league, teams, date, sliders, checkbox), the selected tab, the status line and any simulated result (redrawn without replaying the bar animation). The choice persists in `~/.futbolkronos.json`.

### Named Rules
**The Tonal Step Rule.** To separate a region, step its background one tone or draw a 1 px hairline. Never add a shadow, glow or gradient.

## Shapes

Square-cornered throughout: fields, combobox, buttons, tabs, table rows and the outcome bar segments have no radius. The checkbox is an 18 px square with single-pixel clipped corners. The only round forms are on the slider (3 px track with round caps, 14 px circular thumb, 1.5 px home-blue ring on focus) and the uncertainty band brackets (thin line with short end ticks). Heatmap cells are separated by 2 px surface-coloured gutters.

## Components

### Buttons
- **Shape:** square (0px).
- **Primary** ("Simular partido"): ink fill, ground text, 11pt bold, 10 px × 12 px padding, full sidebar width, hand cursor. Hover goes to pure white; pressed to a softer grey (#C9CFD8); disabled to a dark slate (#2E3440) with faint text; focus draws a home-blue border.
- **Ghost** (export actions): field fill, 1 px hairline border, ink text, 6 px × 8 px padding. Hover to field-hover, pressed to hairline, disabled to sidebar layer with faint text. They act on results, so they live by the tabs, not in the form, and stay disabled until a simulation exists.

### Inputs / Fields
- **Style:** field fill, 1 px hairline border, ink text, 5 px × 8 px padding, square. Comboboxes are read-only selectors with a muted 14 px arrow that turns ink on hover.
- **Focus:** border turns home blue. Hover border lightens (#38404C) and fill moves to field-hover.
- **Disabled:** sidebar-layer fill, faint text, hairline arrow. Drop-down lists use field fill and selection-blue highlight, flat with no border.

### Slider (custom)
Hand-drawn on a canvas: 3 px hairline track, travelled portion and 14 px thumb in muted tones at rest, both turning ink on hover, drag or focus; a home-blue ring marks keyboard focus; arrow keys step. Label left in field-label style, live value right in bold ink, formatted in Spanish ("50.000", "1,00"). Deliberately unanimated: it must track the pointer instantly.

### Checkbox (custom)
18 px image indicator: field fill with a #3A424F border off, field-hover with muted border on hover, home-blue fill with a dark ink-on-light check when selected. Label in ink, turning white on hover.

### Navigation (tabs)
Text-only tabs, 10pt, 8 px × 16 px padding, no borders. Unselected: muted text on ground; hover: ink text on a barely lighter ground (#13161C); selected: ink text on surface, flush with the pane. No underline indicator.

### Tables
Header row 9pt bold muted, flat, 8 px padding. Rows 28 px, surface with zebra-row alternation, selection blue when selected. Text columns left-aligned, numeric columns right-aligned. The top-ranked score row is bold. Thin vertical scrollbar in hairline tones.

### Match Header (signature)
The first thing on screen and the core of the system. Metadata line in faint label type ("España · Primera · dom 01/11/2026 · 50.000 simulaciones"); team names in display type at opposite edges; a full-width 1·X·2 bar split in proportion to the probabilities (home blue, draw grey, away orange, 3 px gaps); above it, when model uncertainty is on, muted brackets showing where each boundary could fall (percentiles 10 to 90); below it, the three percentages in their team colours (draw in ink) with sublabels "Gana X · modelo 31–35 %". A hairline, then the data row: caption in faint label, value in bold title type, note in muted label. The real result value is hit green or miss red with the verdict and the probability the model gave it.

**Motion.** The header owns the system's single animation: the bar morphs from the previous split to the new one. First time it starts from equal thirds with a strong ease-out (quintic, about cubic-bezier(0.23, 1, 0.32, 1)) over 260 ms; afterwards it eases in-out (quartic, about cubic-bezier(0.77, 0, 0.175, 1)) over 280 ms. It is interruptible (a new result restarts from the current position), and the band brackets appear only once the bar has settled. The only other motion is the theme switch's knob.

### Theme Switch
Bottom-right corner, at the end of the disclaimer line, labelled "Modo oscuro" in faint label type; no settings panel. An iOS-style switch: 46 × 28 px pill track, 24 px white knob with a soft 1 px drop. Track is #34C759 (iOS green) when dark mode is on and a neutral grey when off, crossfading as the knob moves. Pressing stretches the knob 6 px toward the centre, as on iOS; releasing slides it with the strong ease-out over 200 ms, and the window changes mode 40 ms after it lands. Space and Enter toggle it; keyboard focus draws a home-blue ring around the track.

### Charts (matplotlib)
Figure and axes on surface; top and right spines removed, remaining spines hairline, zero-length ticks in muted 9pt, chart-grid gridlines behind data. Titles 10.5pt bold, left-aligned, and they name their own encoding ("trazo fino = banda del modelo", "recuadro = real"). Panels: horizontal 1·X·2 bars with band brackets and bold value labels; a 5×5 score heatmap on a field → deep blue → home blue → pale blue ramp with values in-cell (dark ink on the brightest cells), real score outlined in gold; grouped home/away goal bars with a frameless legend; convergence lines on a log axis with end-of-line labels de-collided in team colours. Axis values use "%" with a space and comma decimals.

## Do's and Don'ts

### Do:
- **Do** keep every new surface on the two-layer ground (ground, sidebar-layer) with surface for content panes and field for controls.
- **Do** colour a team only with its side colour (home-blue left, away-orange right, draw-grey between) in every chart, figure and label that refers to it.
- **Do** show the model's band on every 1·X·2 display when uncertainty is enabled (brackets on bars, "modelo a–b %" sublabels).
- **Do** reserve hit-green / miss-red for verdicts (real result, W/L form) and real-gold for marking the real score.
- **Do** separate regions with 1 px hairlines and tonal steps; keep corners square except the slider.
- **Do** format numbers the Spanish way ("33,5 %", "50.000") and keep numeric table columns right-aligned.
- **Do** draw empty states with the same geometry as the filled state, using "—" placeholders in ink-faint.

### Don't:
- **Don't** build rows of identical KPI cards with an indigo accent, or charts styled like a scientific report.
- **Don't** fill the primary button (or any chrome) with a data colour.
- **Don't** add shadows, glows or gradients to separate surfaces.
- **Don't** animate anything other than the 1·X·2 bar morph; sliders and controls respond instantly.
- **Don't** present a probability without its uncertainty context or as a certainty.
