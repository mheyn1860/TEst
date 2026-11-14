"""
Convertible-Bond-Monitor – Web-Oberfläche
=========================================

Start lokal:      streamlit run app.py
Optionaler Login: Umgebungsvariable CB_APP_PASSWORD setzen (oder in .streamlit/secrets.toml
                  `password = "..."`). Ohne Passwort ist die Seite offen.
"""

from __future__ import annotations

import hmac
import io
import os
from datetime import date

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from convertible_tool.monitor import DATA_DIR, load_universe, run
from convertible_tool.pricing import MarketInputs, bond_floor, parity, tf_price

st.set_page_config(page_title="CB Monitor Europa", page_icon="📈", layout="wide")

UNIVERSE_CSV = DATA_DIR / "universe.csv"
QUOTES_CSV = DATA_DIR / "cb_quotes.csv"


# ---------------------------------------------------------------------------
# Login (optional)
# ---------------------------------------------------------------------------
def _configured_password() -> str | None:
    pw = os.environ.get("CB_APP_PASSWORD")
    if pw:
        return pw
    try:
        return st.secrets.get("password")
    except Exception:
        return None


def require_login() -> None:
    pw = _configured_password()
    if not pw or st.session_state.get("auth_ok"):
        return
    st.title("📈 CB Monitor Europa")
    entered = st.text_input("Passwort", type="password")
    if entered:
        if hmac.compare_digest(entered, pw):
            st.session_state["auth_ok"] = True
            st.rerun()
        st.error("Falsches Passwort")
    st.stop()


require_login()


# ---------------------------------------------------------------------------
# Daten
# ---------------------------------------------------------------------------
@st.cache_data(ttl=900, show_spinner="Lade Kurse und bewerte Anleihen …")
def load_data(demo: bool, source: str, rf: float, steps: int, _stamp: float) -> pd.DataFrame:
    return run(demo=demo, source=source, rf=rf, steps=steps, save=not demo)


def fmt_pct(x):
    return "–" if pd.isna(x) else f"{x:,.1f} %"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Einstellungen")
    mode = st.radio("Datenquelle", ["Live (Yahoo + CSV)", "Live (Yahoo + LSEG)", "Demo (synthetisch)"])
    demo = mode.startswith("Demo")
    source = "lseg" if "LSEG" in mode else "csv"
    rf = st.number_input("Risikofreier Zins (p.a.)", value=0.022, step=0.001, format="%.3f")
    steps = st.select_slider("Baumschritte (Genauigkeit)", [100, 200, 300, 500], value=200)
    if st.button("🔄 Jetzt aktualisieren", use_container_width=True):
        st.session_state["stamp"] = pd.Timestamp.now().timestamp()
    stamp = st.session_state.setdefault("stamp", 0.0)

try:
    df = load_data(demo, source, rf, steps, stamp)
except Exception as e:  # Netz-/Lizenzfehler sichtbar machen statt abstürzen
    st.error(f"Daten konnten nicht geladen werden: {e}")
    st.info("Tipp: Mit „Demo“ lässt sich die Oberfläche ohne Datenzugang testen.")
    st.stop()

if df.empty:
    st.warning("Keine Daten – Aktienkurse nicht abrufbar oder Universum leer.")
    st.stop()

with st.sidebar:
    countries = st.multiselect("Land", sorted(df["country"].unique()), default=sorted(df["country"].unique()))
    sectors = st.multiselect("Sektor", sorted(df["sector"].unique()), default=sorted(df["sector"].unique()))
    d_lo, d_hi = st.slider("Delta-Bereich", 0.0, 1.0, (0.0, 1.0), 0.05)

view = df[df["country"].isin(countries) & df["sector"].isin(sectors)
          & df["delta"].between(d_lo - 1e-6, d_hi + 1e-6)].copy()

# ---------------------------------------------------------------------------
# Kopf
# ---------------------------------------------------------------------------
st.title("📈 Convertible-Bond-Monitor Europa")
st.caption(f"Bewertungstag {df['valuation_date'].iloc[0]} · Quelle: {df['data_mode'].iloc[0]} · "
           f"Aktienkurse via Yahoo Finance (verzögert)")
if demo:
    st.warning("DEMO-Modus: Aktien- und CB-Kurse sind synthetisch und nicht real.")
if not df["verified"].all():
    st.info("Stammdaten mit „verified = false“ sind Beispielwerte. Bitte im Tab **Stammdaten** "
            "mit Prospekt/Termsheet abgleichen.")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Anleihen", len(view))
c2.metric("Ø Delta", f"{view['delta'].mean():.2f}" if len(view) else "–")
c3.metric("Ø Prämie", fmt_pct(view["premium_pct"].mean()))
c4.metric("Ø Rendite", fmt_pct(view["ytm_pct"].mean()))
c5.metric("Mit Marktkurs", f"{view['market_clean'].notna().sum()} / {len(view)}")

tab_mon, tab_detail, tab_quotes, tab_static = st.tabs(
    ["Übersicht", "Einzelanalyse", "CB-Kurse erfassen", "Stammdaten"])

# ---------------------------------------------------------------------------
# Übersicht
# ---------------------------------------------------------------------------
with tab_mon:
    cols = {
        "issuer": "Emittent", "country": "Land", "sector": "Sektor", "stock_price": "Aktie",
        "stock_chg_1d_pct": "Δ 1T %", "parity": "Parität", "market_clean": "CB Kurs",
        "model_clean": "Modell", "rich_cheap_pct": "rich/cheap %", "premium_pct": "Prämie %",
        "bond_floor": "Floor", "invest_premium_pct": "Inv.-Prämie %", "ytm_pct": "YTM %",
        "delta": "Delta", "gamma": "Gamma", "vega": "Vega", "credit_sens_10bp": "CS +10bp",
        "implied_vol": "impl. Vol", "hist_vol": "hist. Vol", "maturity": "Fälligkeit",
        "coupon": "Kupon", "conv_price": "Wandlungspreis",
    }
    table = view[list(cols)].rename(columns=cols)
    table["impl. Vol"] *= 100
    table["hist. Vol"] *= 100
    num = {c: st.column_config.NumberColumn(format="%.2f") for c in table.columns
           if pd.api.types.is_numeric_dtype(table[c])}
    st.dataframe(table, hide_index=True, use_container_width=True, column_config=num, height=520)

    left, right = st.columns(2)
    with left:
        st.subheader("Delta vs. Wandlungsprämie")
        chart_df = view.assign(premium=view["premium_pct"].fillna(view["model_clean"] / view["parity"] * 100 - 100))
        st.altair_chart(
            alt.Chart(chart_df).mark_circle(size=120).encode(
                x=alt.X("delta:Q", title="Delta", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("premium:Q", title="Prämie % (Markt, sonst Modell)"),
                color=alt.Color("sector:N", title="Sektor"),
                tooltip=["issuer", alt.Tooltip("delta:Q", format=".2f"),
                         alt.Tooltip("premium:Q", format=".1f"), "maturity"],
            ).interactive(), use_container_width=True)
    with right:
        st.subheader("Rich / Cheap vs. Modell")
        rc = view.dropna(subset=["rich_cheap_pct"])
        if rc.empty:
            st.caption("Erst verfügbar, wenn CB-Marktkurse vorliegen (Tab „CB-Kurse erfassen“).")
        else:
            st.altair_chart(
                alt.Chart(rc).mark_bar().encode(
                    x=alt.X("rich_cheap_pct:Q", title="Markt vs. Modell %"),
                    y=alt.Y("issuer:N", sort="-x", title=None),
                    color=alt.condition("datum.rich_cheap_pct > 0", alt.value("#d1495b"), alt.value("#2e86ab")),
                    tooltip=["issuer", alt.Tooltip("rich_cheap_pct:Q", format=".2f")],
                ), use_container_width=True)

    buf = io.BytesIO()
    view.to_excel(buf, index=False)
    d1, d2 = st.columns(2)
    d1.download_button("⬇ Excel", buf.getvalue(), f"cb_monitor_{date.today()}.xlsx", use_container_width=True)
    d2.download_button("⬇ CSV", view.to_csv(index=False).encode(), f"cb_monitor_{date.today()}.csv",
                       use_container_width=True)

# ---------------------------------------------------------------------------
# Einzelanalyse
# ---------------------------------------------------------------------------
with tab_detail:
    bonds = {b.id: b for b in load_universe()}
    pick = st.selectbox("Anleihe", view["id"], format_func=lambda i: f"{bonds[i].issuer} ({i})")
    if pick:
        row = view.set_index("id").loc[pick]
        b = bonds[pick]
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Aktie", f"{row.stock_price:,.2f}", fmt_pct(row.stock_chg_1d_pct))
        m2.metric("CB Kurs", "–" if pd.isna(row.market_clean) else f"{row.market_clean:.2f}")
        m3.metric("Modell", f"{row.model_clean:.2f}")
        m4.metric("Parität", f"{row.parity:.2f}")
        m5.metric("Delta", f"{row.delta:.2f}")
        m6.metric("impl. Vol", "–" if pd.isna(row.implied_vol) else f"{row.implied_vol*100:.1f} %")

        vol = st.slider("Volatilität für Profil", 0.05, 0.9, float(row.hist_vol), 0.01)
        S0 = float(row.stock_price)
        grid = np.linspace(S0 * 0.3, S0 * 2.0, 35)
        val_date = date.fromisoformat(row.valuation_date)
        prof = []
        for S in grid:
            mk = MarketInputs(stock_price=S, volatility=vol, risk_free=rf, valuation_date=val_date)
            prof += [{"Aktie": S, "Kurve": "Modellwert", "Wert": tf_price(b, mk, 150)},
                     {"Aktie": S, "Kurve": "Parität", "Wert": parity(b, S)},
                     {"Aktie": S, "Kurve": "Bond Floor", "Wert": bond_floor(b, mk)}]
        prof = pd.DataFrame(prof)
        base = alt.Chart(prof).mark_line().encode(
            x=alt.X("Aktie:Q", title="Aktienkurs"),
            y=alt.Y("Wert:Q", title="Wert pro 100 Nominal"),
            color=alt.Color("Kurve:N"),
            strokeDash=alt.condition("datum.Kurve == 'Modellwert'", alt.value([1, 0]), alt.value([5, 4])))
        rule = alt.Chart(pd.DataFrame({"x": [S0]})).mark_rule(color="gray").encode(x="x:Q")
        layers = base + rule
        if not pd.isna(row.market_clean):
            layers += alt.Chart(pd.DataFrame({"x": [S0], "y": [row.market_clean]})).mark_point(
                size=150, filled=True, color="black").encode(x="x:Q", y="y:Q", tooltip=["y:Q"])
        st.altair_chart(layers.interactive(), use_container_width=True)
        st.caption("Graue Linie = aktueller Aktienkurs, schwarzer Punkt = CB-Marktkurs.")

        st.json({k: (v.isoformat() if isinstance(v, date) else v) for k, v in b.__dict__.items()},
                expanded=False)

# ---------------------------------------------------------------------------
# CB-Kurse erfassen
# ---------------------------------------------------------------------------
with tab_quotes:
    st.write("Aktuelle Clean-Preise der Wandelanleihen eintragen (z. B. aus Bloomberg, Broker-Run, "
             "Börse Frankfurt/Stuttgart) oder eine CSV hochladen (Spalten `id` oder `isin`, `clean_price`).")
    uni = pd.read_csv(UNIVERSE_CSV, dtype={"isin": str})
    existing = pd.read_csv(QUOTES_CSV) if QUOTES_CSV.exists() else pd.DataFrame(columns=["id", "clean_price"])
    latest = existing.dropna(subset=["id"]).drop_duplicates("id", keep="last").set_index("id")["clean_price"] \
        if "id" in existing.columns and not existing.empty else pd.Series(dtype=float)
    edit = uni[["id", "issuer", "isin"]].copy()
    edit["clean_price"] = edit["id"].map(latest)
    edited = st.data_editor(edit, hide_index=True, use_container_width=True,
                            disabled=["id", "issuer", "isin"],
                            column_config={"clean_price": st.column_config.NumberColumn("CB Kurs (clean)", format="%.3f")})
    up = st.file_uploader("…oder CSV hochladen", type="csv")
    if st.button("💾 Kurse speichern", type="primary"):
        new = pd.read_csv(up) if up else edited[["id", "isin", "clean_price"]]
        new = new.dropna(subset=["clean_price"]).assign(date=date.today().isoformat())
        pd.concat([existing, new], ignore_index=True).to_csv(QUOTES_CSV, index=False)
        st.session_state["stamp"] = pd.Timestamp.now().timestamp()
        st.success(f"{len(new)} Kurse gespeichert – Übersicht wird neu berechnet.")
        st.rerun()

# ---------------------------------------------------------------------------
# Stammdaten
# ---------------------------------------------------------------------------
with tab_static:
    st.write("Bond-Universum bearbeiten: Zeilen hinzufügen/löschen, Konditionen prüfen und "
             "`verified` setzen. Yahoo-Ticker z. B. `SAP.DE`, `SU.PA`, `PRY.MI`, `CLNX.MC`.")
    uni_edit = st.data_editor(pd.read_csv(UNIVERSE_CSV, dtype={"isin": str}),
                              num_rows="dynamic", hide_index=True, use_container_width=True)
    if st.button("💾 Stammdaten speichern", type="primary"):
        uni_edit.to_csv(UNIVERSE_CSV, index=False)
        st.session_state["stamp"] = pd.Timestamp.now().timestamp()
        st.success("Gespeichert.")
        st.rerun()

st.caption("Modell: Binomialbaum (Tsiveriotis-Fernandes) mit Call/Put, flache Zins-/Spreadkurve. "
           "Keine Anlageberatung.")
