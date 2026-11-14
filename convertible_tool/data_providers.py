"""
Datenquellen
============

Aktienkurse & Volatilität:
  * YahooEquityProvider  – kostenlos via yfinance (verzögert, ~15 min)
  * DemoEquityProvider   – synthetische Daten zum Offline-Testen

Convertible-Bond-Kurse (es gibt KEINE freie Echtzeit-API für CB-Kurse):
  * CsvQuoteProvider     – Kurse aus data/cb_quotes.csv (Export aus Bloomberg,
                           LSEG, Broker-Run, Börse Frankfurt/Stuttgart …)
  * LsegQuoteProvider    – LSEG Data Library (Workspace-Lizenz nötig)
  * DemoQuoteProvider    – synthetische Kurse (nur Test!)

Eigene Provider: Klasse mit `get_quotes(bonds) -> dict[id, clean_price]`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd


@dataclass
class EquitySnapshot:
    price: float
    hist_vol_1y: float
    currency: str = ""
    change_1d_pct: float = float("nan")


class EquityProvider(Protocol):
    def get(self, tickers: list[str]) -> dict[str, EquitySnapshot]: ...


class QuoteProvider(Protocol):
    def get_quotes(self, bonds) -> dict[str, float]: ...


# ---------------------------------------------------------------------------
# Aktien
# ---------------------------------------------------------------------------
def _hist_vol(close: pd.Series) -> float:
    rets = np.log(close / close.shift(1)).dropna()
    return float(rets.std() * math.sqrt(252)) if len(rets) > 20 else float("nan")


class YahooEquityProvider:
    def get(self, tickers: list[str]) -> dict[str, EquitySnapshot]:
        import yfinance as yf

        data = yf.download(tickers, period="1y", interval="1d", auto_adjust=True,
                           progress=False, group_by="ticker", threads=True,
                           timeout=15)
        out: dict[str, EquitySnapshot] = {}
        for t in tickers:
            try:
                close = (data[t]["Close"] if len(tickers) > 1 else data["Close"]).dropna()
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                if close.empty:
                    continue
                out[t] = EquitySnapshot(
                    price=float(close.iloc[-1]),
                    hist_vol_1y=_hist_vol(close),
                    change_1d_pct=float((close.iloc[-1] / close.iloc[-2] - 1) * 100)
                    if len(close) > 1 else float("nan"),
                )
            except (KeyError, IndexError):
                continue
        return out


class DemoEquityProvider:
    """Deterministische Pseudo-Kurse rund um den Wandlungspreis."""

    def __init__(self, bonds, seed: int = 7):
        self._ref = {b.stock_ticker: b.conversion_price for b in bonds}
        self._rng = np.random.default_rng(seed)

    def get(self, tickers: list[str]) -> dict[str, EquitySnapshot]:
        out = {}
        for t in tickers:
            ref = self._ref.get(t, 50.0)
            out[t] = EquitySnapshot(
                price=round(ref * self._rng.uniform(0.55, 1.35), 2),
                hist_vol_1y=float(self._rng.uniform(0.22, 0.50)),
                change_1d_pct=float(self._rng.normal(0, 1.5)),
            )
        return out


# ---------------------------------------------------------------------------
# CB-Kurse
# ---------------------------------------------------------------------------
class CsvQuoteProvider:
    """Liest data/cb_quotes.csv mit Spalten: id ODER isin, clean_price[, date].
    Bei mehreren Zeilen je Bond gewinnt die jüngste."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def get_quotes(self, bonds) -> dict[str, float]:
        if not self.path.exists():
            return {}
        df = pd.read_csv(self.path)
        if "date" in df.columns:
            df = df.sort_values("date")
        by_id = dict(zip(df.get("id", pd.Series(dtype=str)).astype(str), df["clean_price"])) \
            if "id" in df.columns else {}
        by_isin = dict(zip(df["isin"].astype(str), df["clean_price"])) if "isin" in df.columns else {}
        out = {}
        for b in bonds:
            px = by_id.get(b.id, by_isin.get(b.isin) if b.isin else None)
            if px is not None and not pd.isna(px):
                out[b.id] = float(px)
        return out


class LsegQuoteProvider:
    """LSEG Data Library (pip install lseg-data, Workspace/Desktop-Session nötig).
    Ungetestet ohne Lizenz – Feldnamen ggf. an deinen Zugang anpassen."""

    def __init__(self, field: str = "TR.BIDPRICE"):
        self.field = field

    def get_quotes(self, bonds) -> dict[str, float]:
        import lseg.data as ld

        isins = [b.isin for b in bonds if b.isin]
        if not isins:
            return {}
        ld.open_session()
        try:
            df = ld.get_data(universe=isins, fields=[self.field])
        finally:
            ld.close_session()
        price_by_isin = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
        return {b.id: float(price_by_isin[b.isin]) for b in bonds
                if b.isin in price_by_isin and not pd.isna(price_by_isin[b.isin])}


class DemoQuoteProvider:
    """Synthetische Marktpreise = Modellpreis ± Rauschen. NUR zum Testen."""

    def __init__(self, model_prices: dict[str, float], seed: int = 11):
        self._m = model_prices
        self._rng = np.random.default_rng(seed)

    def get_quotes(self, bonds) -> dict[str, float]:
        return {b.id: round(self._m[b.id] * self._rng.uniform(0.97, 1.03), 2)
                for b in bonds if b.id in self._m}
