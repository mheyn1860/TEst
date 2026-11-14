"""
Convertible-Bond-Monitor – Einstiegspunkt
=========================================

  python -m convertible_tool.monitor                 # live: Yahoo + data/cb_quotes.csv
  python -m convertible_tool.monitor --demo          # offline, synthetische Daten
  python -m convertible_tool.monitor --source lseg   # CB-Kurse aus LSEG
  python -m convertible_tool.monitor --rf 0.021 --ids DHER_2030 ZAL_2029

Ausgabe:
  * Tabelle im Terminal
  * output/cb_monitor_YYYY-MM-DD.xlsx und .csv
  * Historie in convertible_tool/data/history.csv (für Zeitreihen / Charts)
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from .data_providers import (CsvQuoteProvider, DemoEquityProvider, DemoQuoteProvider,
                             LsegQuoteProvider, YahooEquityProvider)
from .pricing import ConvertibleBond, MarketInputs, analyse, tf_price

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path("output")


def _d(x):
    return None if pd.isna(x) or x == "" else date.fromisoformat(str(x))


def load_universe(path: Path = DATA_DIR / "universe.csv") -> list[ConvertibleBond]:
    df = pd.read_csv(path, dtype={"isin": str}).fillna({"isin": ""})
    bonds = []
    for r in df.to_dict("records"):
        bonds.append(ConvertibleBond(
            id=r["id"], issuer=r["issuer"], isin=r["isin"], stock_ticker=r["stock_ticker"],
            currency=r["currency"], country=r["country"], sector=r["sector"],
            coupon=float(r["coupon"]), coupon_freq=int(r["coupon_freq"]),
            maturity=_d(r["maturity"]), conversion_price=float(r["conversion_price"]),
            redemption=float(r["redemption"]), credit_spread_bp=float(r["credit_spread_bp"]),
            dividend_yield=float(r["dividend_yield"]),
            call_date=_d(r["call_date"]),
            call_price=float(r["call_price"]) if not pd.isna(r["call_price"]) else 100.0,
            put_date=_d(r["put_date"]),
            put_price=float(r["put_price"]) if not pd.isna(r["put_price"]) else 100.0,
            verified=str(r["verified"]).lower() == "true",
        ))
    return bonds


def run(demo: bool = False, source: str = "csv", rf: float = 0.022, ids=None,
        steps: int = 300, val_date: date | None = None, save: bool = True) -> pd.DataFrame:
    val_date = val_date or date.today()
    bonds = [b for b in load_universe() if b.maturity > val_date]
    if ids:
        bonds = [b for b in bonds if b.id in ids]

    tickers = sorted({b.stock_ticker for b in bonds})
    eq_provider = DemoEquityProvider(bonds) if demo else YahooEquityProvider()
    equities = eq_provider.get(tickers)

    missing = [t for t in tickers if t not in equities]
    if missing:
        print(f"⚠ Keine Aktienkurse für: {', '.join(missing)}", file=sys.stderr)
    bonds = [b for b in bonds if b.stock_ticker in equities]

    def mkt_for(b):
        e = equities[b.stock_ticker]
        vol = e.hist_vol_1y if e.hist_vol_1y == e.hist_vol_1y else 0.30
        return MarketInputs(stock_price=e.price, volatility=vol, risk_free=rf, valuation_date=val_date)

    if demo:
        quotes = DemoQuoteProvider({b.id: tf_price(b, mkt_for(b), 150) for b in bonds}).get_quotes(bonds)
    elif source == "lseg":
        quotes = LsegQuoteProvider().get_quotes(bonds)
    else:
        quotes = CsvQuoteProvider(DATA_DIR / "cb_quotes.csv").get_quotes(bonds)

    rows = []
    for b in bonds:
        res = analyse(b, mkt_for(b), quotes.get(b.id), steps=steps)
        res["stock_chg_1d_pct"] = equities[b.stock_ticker].change_1d_pct
        res["country"], res["sector"] = b.country, b.sector
        res["valuation_date"] = val_date.isoformat()
        res["data_mode"] = "DEMO" if demo else source.upper()
        rows.append(res)
    df = pd.DataFrame(rows)

    if save and not df.empty:
        OUT_DIR.mkdir(exist_ok=True)
        stem = OUT_DIR / f"cb_monitor_{val_date.isoformat()}{'_DEMO' if demo else ''}"
        df.to_csv(f"{stem}.csv", index=False)
        try:
            df.to_excel(f"{stem}.xlsx", index=False)
        except ImportError:
            pass
        if not demo:
            # Historie: pro Tag und Bond nur der letzte Stand
            hist = DATA_DIR / "history.csv"
            all_rows = pd.concat([pd.read_csv(hist), df]) if hist.exists() else df
            all_rows.drop_duplicates(["valuation_date", "id"], keep="last").to_csv(hist, index=False)
    return df


def print_table(df: pd.DataFrame) -> None:
    cols = {
        "issuer": "Emittent", "stock_price": "Aktie", "stock_chg_1d_pct": "Δ1T%",
        "parity": "Parität", "market_clean": "CB Kurs", "model_clean": "Modell",
        "rich_cheap_pct": "rich/cheap%", "premium_pct": "Prämie%", "bond_floor": "Floor",
        "ytm_pct": "YTM%", "delta": "Delta", "implied_vol": "impl.Vol", "hist_vol": "hist.Vol",
    }
    view = df[list(cols)].rename(columns=cols)
    with pd.option_context("display.width", 200, "display.max_columns", 30,
                           "display.float_format", "{:.2f}".format):
        print(view.to_string(index=False))
    if not df["verified"].all():
        print("\n⚠ Stammdaten mit verified=false sind BEISPIELWERTE – vor Nutzung mit "
              "Prospekt/Termsheet abgleichen (convertible_tool/data/universe.csv).")
    if df["market_clean"].isna().any():
        print("ℹ Ohne CB-Marktkurs nur Modellwerte. Kurse in convertible_tool/data/cb_quotes.csv "
              "eintragen oder --source lseg nutzen.")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Convertible-Bond-Monitor (Europa)")
    ap.add_argument("--demo", action="store_true", help="Offline mit synthetischen Daten")
    ap.add_argument("--source", choices=["csv", "lseg"], default="csv", help="Quelle für CB-Kurse")
    ap.add_argument("--rf", type=float, default=0.022, help="Risikofreier Zins (stetig)")
    ap.add_argument("--ids", nargs="*", help="Nur diese Bond-IDs")
    ap.add_argument("--steps", type=int, default=300, help="Baumschritte")
    args = ap.parse_args(argv)

    df = run(demo=args.demo, source=args.source, rf=args.rf, ids=args.ids, steps=args.steps)
    if df.empty:
        print("Keine Daten.")
        return 1
    print_table(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
