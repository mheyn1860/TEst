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

## Web-Oberfläche

```bash
streamlit run app.py                      # http://localhost:8501
CB_APP_PASSWORD=geheim streamlit run app.py   # mit Login
```

Tabs: **Übersicht** (Tabelle, Filter, Delta-vs-Prämie, Rich/Cheap, Excel/CSV-Download),
**Einzelanalyse** (Kennzahlen + Payoff-Profil Modellwert/Parität/Bond Floor mit Vola-Regler),
**CB-Kurse erfassen** (Kurse eintippen oder CSV hochladen), **Stammdaten** (Universum bearbeiten).

### Als eigene Website veröffentlichen
- **Streamlit Community Cloud** (kostenlos): Repo verbinden, `app.py` als Einstiegspunkt,
  Passwort unter *Settings → Secrets* als `password = "..."`. Optional App auf „privat“ stellen.
- **Eigener Server / Cloud (Docker)**:
  `docker build -t cb-monitor . && docker run -e CB_APP_PASSWORD=... -p 8501:8501 cb-monitor`
  (z. B. auf Render, Fly.io, Azure App Service, Hetzner) – davor HTTPS über den Hoster oder einen Reverse-Proxy.

Hinweis: Auf Hostern ohne persistente Festplatte gehen eingetragene Kurse/Stammdaten bei
Neustarts verloren – dann CSVs im Repo pflegen oder eine Datenbank anbinden.
