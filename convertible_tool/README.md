# Convertible-Bond-Monitor (Europa)

Auf Knopfdruck: Aktienkurse, CB-Kurse und alle relevanten Kennzahlen für europäische Wandelanleihen.

```bash
pip install -r requirements.txt
python -m convertible_tool.monitor --demo          # Offline-Test mit synthetischen Daten
python -m convertible_tool.monitor                 # Live: Aktien via Yahoo, CB-Kurse aus data/cb_quotes.csv
python -m convertible_tool.monitor --source lseg   # CB-Kurse via LSEG Data Library (Lizenz nötig)
```

## Kennzahlen je Bond
Aktienkurs, Δ1T, Parität, CB-Kurs, theoretischer Wert (Binomialbaum Tsiveriotis-Fernandes
mit Call/Put), rich/cheap, Wandlungsprämie, Bond Floor, Investment-Prämie, YTM, Delta, Gamma,
Vega, Credit-Sensitivität (+10 bp), implizite & historische Volatilität.

Ausgabe: Terminal-Tabelle, `output/cb_monitor_<datum>.xlsx/.csv`, Historie in `data/history.csv`.

## Daten
| Was | Quelle |
|---|---|
| Aktienkurse, hist. Vola | Yahoo Finance (`yfinance`, kostenlos, verzögert) |
| CB-Kurse | `data/cb_quotes.csv` (Spalten `id` oder `isin`, `clean_price`, `date`) oder LSEG |
| Stammdaten | `data/universe.csv` |

**Wichtig:** Die Stammdaten in `universe.csv` sind Beispielwerte (`verified=false`).
Kupon, Wandlungspreis, Fälligkeit, Call/Put und ISIN vor Nutzung mit Prospekt/Termsheet abgleichen
und dann `verified=true` setzen. Credit Spreads sind Schätzwerte.

## Modellannahmen
Flache Zins- und Spreadkurve, stetige Dividendenrendite, Aktie und Anleihe in gleicher Währung,
Soft-Call als harter Call ab Call-Datum. Keine Anlageberatung.
