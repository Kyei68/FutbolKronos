# Product

<!-- impeccable:product-schema 1 -->

## Platform

desktop (Tkinter, Windows). Not web, not native mobile.

## Stack

delegated: stays in Tkinter + ttk + matplotlib (TkAgg) in `futbolkronos_gui.py`. No new dependencies; it keeps opening with `Abrir FutbolKronos.bat`. The user stated no preference.

## Users

One user: the author of the model. They use FutbolKronos as a research tool to **evaluate how well the model predicts**: they simulate upcoming or already-played matches, compare the prediction with the real result and with bookmaker odds, and judge whether the model is calibrated.

## Product Purpose

Simulate a football match 10,000–100,000 times with the FutbolKronos model (a Kronos transformer adapted to football) and show the resulting probability distribution: 1X2, exact scores, goals per team, markets (over/under, both teams to score, double chance), convergence and the model's uncertainty band (dropout, percentiles 10–90). If the match has already been played, it simulates using only prior data and compares against the real result.

## Positioning

Not a betting tipster: it is a model lab. The value is seeing the model's full distribution and uncertainty, and contrasting it with the market and with reality.

## Operating Context

- Leagues from football-data (SP1 and others) and European competitions (CL, EL, ECL and their qualifiers).
- "Upcoming matches" list for the next two matchdays per competition.
- Optional 1·X·2 odds; when left empty, published odds are used if available.
- Exports: chart PNG and score-table CSV to `resultados/`.
- Interface copy in Spanish (Spain), with comma decimals and dot thousands separators.

## Capabilities and Constraints

- Simulation parameters: number of simulations (10k–100k), temperature (0.5–1.5), toggle for model uncertainty.
- Context: recent form, Elo of each team before the match, market without margin, notice when there is little history.
- Work runs in a background thread with a message queue; the UI must not block.
- Must not alter the logic of `simulador.py` or the computed results.

## Brand Commitments

Name: FutbolKronos. No logo or confirmed brand assets.

## Evidence on Hand

Real data from the model and the leagues. No testimonials or commercial claims; none should be invented.

## Product Principles

1. Honesty about uncertainty: probabilities are never presented as certainties; the model's band and the comparison with the market stay visible.
2. The real result is the judge: when a match has already been played, the comparison with reality is the most important element on screen.
3. Scannability over decoration: it is a working tool that gets opened over and over.
