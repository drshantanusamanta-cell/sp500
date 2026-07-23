# -*- coding: utf-8 -*-
"""
================================================================================
 Shantanu's Options Analysis Dashboard  —  Streamlit Edition
 Data: Yahoo Finance (yfinance) live options chain | Demo Mode fallback
 Built as a structural / feature port of the NIFTY options dashboard, adapted
 to a free-data-only pipeline for a US-listed ETF proxy of a CME future.
 Bias Score: -100 to +100 | Regime Classifier | Strategy Engine | OI Momentum
================================================================================

IMPORTANT — READ BEFORE USE
----------------------------
This dashboard tracks S&P 500 (ES1! E-mini S&P 500 futures) via the free, delayed options chain of
SPY (SPDR S&P 500 ETF Trust, the world's most liquid free-data-accessible proxy for S&P 500 futures), NOT a direct live feed of the CME future itself.
There is no free, live, real-time options-chain (OI/Greeks) data source for
CME futures products (ES/NQ/GC/CL/6E and their micros) anywhere — that data
is sold by CME/ICE/Cboe/vendors. SPY is the most liquid, free-data-
accessible proxy available for this underlying.

Yahoo Finance option-chain data is exchange-delayed (typically ~15 minutes)
and open interest updates once per session (it does not tick live like price).
Treat every metric here as directionally useful for paper trading / study,
NOT as an execution-grade or investment-advice signal.
"""

import os, json, time, warnings, math, threading
from datetime import datetime, timedelta
import pytz

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm
import plotly.graph_objs as go
import plotly.io as _pio

import streamlit as st
from streamlit_autorefresh import st_autorefresh

try:
    import yfinance as yf
except Exception:
    yf = None

warnings.filterwarnings("ignore")

# ─── Asset configuration (this block is the ONLY thing that differs between
#     the 4 dashboards in this family — everything else below is generic) ───
ASSET_CONFIG = {
    "key": 'SP500',
    "ticker": 'SPY',
    "short_name": 'S&P 500 / SPY',
    "icon": '📈',
    "underlying_desc": 'SPDR S&P 500 ETF Trust (SPY) — ETF proxy for E-mini S&P 500 futures (ES1!)',
    "multiplier": 100,
    "currency": '$',
    "demo_spot": 560.0,
    "risk_free_rate": 0.05,
}

MULTIPLIER      = ASSET_CONFIG["multiplier"]
RISK_FREE_RATE  = ASSET_CONFIG.get("risk_free_rate", 0.05)
TICKER          = ASSET_CONFIG["ticker"]

# Force explicit colours for all charts — fixes white labels on iOS Safari / iPad
_pio.templates["_mobile_fix"] = go.layout.Template(
    layout=go.Layout(
        font=dict(color="#1A1A2E"),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#F9FAFB",
        xaxis=dict(tickfont=dict(color="#1A1A2E"), title_font=dict(color="#1A1A2E"),
                   linecolor="#E5E7EB", gridcolor="#F3F4F6"),
        yaxis=dict(tickfont=dict(color="#1A1A2E"), title_font=dict(color="#1A1A2E"),
                   linecolor="#E5E7EB", gridcolor="#F3F4F6"),
        legend=dict(font=dict(color="#1A1A2E")),
        hoverlabel=dict(bgcolor="#ffffff", font_color="#1A1A2E"),
        title=dict(font=dict(color="#1A1A2E")),
    )
)
_pio.templates.default = "plotly+_mobile_fix"

# ─── Timezone (all four ETF proxies trade on US exchanges: NYSE Arca / Cboe BZX) ──
MARKET_TZ = pytz.timezone("America/New_York")


def now_et():
    return datetime.now(MARKET_TZ)


def et_str(fmt="%d-%m-%Y  %H:%M:%S ET"):
    return now_et().strftime(fmt)


def is_market_hours():
    n = now_et()
    return n.weekday() < 5 and (9, 30) <= (n.hour, n.minute) <= (16, 0)


# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=f"Shantanu's Options Dashboard — {ASSET_CONFIG['short_name']}",
    page_icon=ASSET_CONFIG["icon"],
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Black-Scholes engine (identical formulas to the NIFTY dashboard) ────────
def _bs_price(S, K, T, r, sigma, opt):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt == "CE":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _bs_greeks(S, K, T, r, sigma, opt):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0, 0.0, 0.0, 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    nd1 = norm.pdf(d1)
    delta = norm.cdf(d1) if opt == "CE" else -norm.cdf(-d1)
    gamma = nd1 / (S * sigma * np.sqrt(T))
    if opt == "CE":
        theta = (-(S * nd1 * sigma) / (2 * np.sqrt(T)) -
                 r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta = (-(S * nd1 * sigma) / (2 * np.sqrt(T)) +
                 r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    vega = S * nd1 * np.sqrt(T) / 100
    return delta, gamma, theta, vega


def _solve_iv(mkt_price, S, K, T, r, opt):
    if T <= 0 or mkt_price <= 0 or S <= 0 or K <= 0:
        return np.nan
    try:
        return brentq(
            lambda v: (_bs_price(S, K, T, r, v, opt) - mkt_price),
            1e-4, 5.0, xtol=1e-5, maxiter=100
        )
    except Exception:
        return np.nan


def safe_num(x, d=0.0):
    try:
        if x is None:
            return d
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return d
        return v
    except (TypeError, ValueError):
        return d


# ─── GEX + IV Rank + Gamma Regime (ported from the NIFTY engine) ─────────────
def compute_true_gex(df, spot):
    """GEX = OI × Gamma × Multiplier × Spot² × 0.01. Calls add, puts subtract."""
    if df is None or df.empty:
        return 0.0, pd.Series(dtype=float), None
    strikes = df["strike"].values
    call_arr = df["call_oi"].values * df["call_gamma"].values * MULTIPLIER * (spot ** 2) * 0.01
    put_arr = df["put_oi"].values * df["put_gamma"].values * MULTIPLIER * (spot ** 2) * 0.01
    net_arr = call_arr - put_arr
    total_gex = float(net_arr.sum())
    gex_series = pd.Series(net_arr, index=strikes)
    cumulative = gex_series.sort_index().cumsum()
    flip_cands = cumulative[cumulative <= 0].index
    gamma_flip = float(flip_cands[-1]) if len(flip_cands) > 0 else None
    return total_gex, gex_series, gamma_flip


def compute_gamma_flip_true(df, spot, T, r=RISK_FREE_RATE, band_pct=0.20, n_pts=60):
    """Price-domain zero-gamma level: recompute gamma at a grid of hypothetical
    spot levels (today's IV smile held fixed per strike) and find the sign
    change of total dealer GEX."""
    if df is None or df.empty or spot <= 0 or T <= 0:
        return None
    lo = max(spot * (1 - band_pct), float(df["strike"].min()))
    hi = min(spot * (1 + band_pct), float(df["strike"].max()))
    if hi <= lo:
        return None
    levels = np.linspace(lo, hi, n_pts)
    strikes = df["strike"].values
    iv_c = df["call_iv"].values / 100.0
    iv_p = df["put_iv"].values / 100.0
    call_oi = df["call_oi"].values
    put_oi = df["put_oi"].values

    total_gamma = np.empty(n_pts)
    for i, S in enumerate(levels):
        cg = np.array([_bs_greeks(S, K, T, r, s, "CE")[1] for K, s in zip(strikes, iv_c)])
        pg = np.array([_bs_greeks(S, K, T, r, s, "PE")[1] for K, s in zip(strikes, iv_p)])
        call_gex = (call_oi * cg * MULTIPLIER * (S ** 2) * 0.01).sum()
        put_gex = (put_oi * pg * MULTIPLIER * (S ** 2) * 0.01).sum()
        total_gamma[i] = call_gex - put_gex

    cross = np.where(np.diff(np.sign(total_gamma)))[0]
    if len(cross) == 0:
        return None
    i = cross[0]
    x0, y0 = levels[i], total_gamma[i]
    x1, y1 = levels[i + 1], total_gamma[i + 1]
    if y1 == y0:
        return float(x0)
    return float(x1 - (x1 - x0) * y1 / (y1 - y0))


def compute_iv_rank(df, atm):
    """Cross-sectional smile-position rank (0-100), NOT a temporal rank."""
    if df is None or df.empty:
        return 0.0, 0.0
    avg_iv = ((df["call_iv"].replace(0, np.nan) + df["put_iv"].replace(0, np.nan)) / 2).dropna()
    if avg_iv.empty:
        return 0.0, 0.0
    row = df[df["strike"] == atm]
    if row.empty:
        row = df.iloc[[(df["strike"] - atm).abs().idxmin()]]
    _c = safe_num(row["call_iv"].iloc[0])
    _p = safe_num(row["put_iv"].iloc[0])
    if _c > 0 or _p > 0:
        atm_iv = (_c + _p) / max(1, (_c > 0) + (_p > 0))
    else:
        atm_iv = 0.0
    iv_min, iv_max = float(avg_iv.min()), float(avg_iv.max())
    if iv_max <= iv_min:
        return 0.0, 0.0
    iv_rank = round((atm_iv - iv_min) / (iv_max - iv_min) * 100, 1)
    iv_pct = round(float((avg_iv <= atm_iv).mean()) * 100, 1)
    return iv_rank, iv_pct


TEMPORAL_IVR_LOOKBACK_DAYS = 20
TEMPORAL_IVR_MIN_SAMPLES = 8
_IV_HISTORY_FILE = f"{ASSET_CONFIG['key'].lower()}_iv_history.json"


def _load_iv_history():
    try:
        if os.path.exists(_IV_HISTORY_FILE):
            with open(_IV_HISTORY_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return []


def _save_iv_history(hist):
    try:
        hist = hist[-500:]
        with open(_IV_HISTORY_FILE, "w") as f:
            json.dump(hist, f)
    except Exception:
        pass  # Streamlit Cloud filesystem is ephemeral — best-effort only


def compute_temporal_iv_rank(current_atm_iv, history):
    if current_atm_iv is None or current_atm_iv <= 0 or not history or len(history) < 2:
        return None, False
    by_date = {}
    for h in history:
        iv = safe_num(h.get("atm_iv", 0))
        if iv <= 0:
            continue
        ts = h.get("ts", "")
        d = ts.split("T")[0] if "T" in ts else ts[:10]
        if d:
            by_date[d] = iv
    if len(by_date) < TEMPORAL_IVR_MIN_SAMPLES:
        return None, False
    sorted_dates = sorted(by_date.keys())
    window_dates = sorted_dates[-TEMPORAL_IVR_LOOKBACK_DAYS:]
    window_ivs = [by_date[d] for d in window_dates]
    iv_min, iv_max = min(window_ivs), max(window_ivs)
    if iv_max - iv_min < 0.5:
        return None, False
    rank = round((current_atm_iv - iv_min) / (iv_max - iv_min) * 100, 1)
    return max(0.0, min(100.0, rank)), True


def classify_gamma_regime(gex, wall_width, momentum, iv_rank, spot, gamma_flip):
    spot_pct = lambda abs_pts: (abs_pts / max(spot, 1)) * 100.0
    flip_dist = abs(spot - gamma_flip) if gamma_flip is not None else 9e9
    _strike_step = max(wall_width / 20, spot * 0.002) if wall_width > 0 else spot * 0.002
    near_flip = flip_dist < max(2.0 * _strike_step, spot * 0.004) if gamma_flip is not None else False
    if iv_rank >= 70:
        vol_regime = "HIGH_VOL"
    elif iv_rank <= 30:
        vol_regime = "LOW_VOL"
    else:
        vol_regime = "MID_VOL"
    wall_width_pct = spot_pct(wall_width)
    if gex > 0 and wall_width_pct <= 1.3 and vol_regime == "LOW_VOL":
        return "PINNED / RANGE", vol_regime, near_flip
    elif gex > 0 and wall_width_pct <= 1.7:
        return "RANGE / PIN", vol_regime, near_flip
    elif gex < 0 and abs(momentum) > (spot * 0.002) and vol_regime in ("MID_VOL", "HIGH_VOL"):
        return "TREND / EXPANSION", vol_regime, near_flip
    elif near_flip:
        return "FLIP ZONE / UNSTABLE", vol_regime, near_flip
    else:
        return "TRANSITION", vol_regime, near_flip


def compute_max_pain(df):
    if df is None or df.empty:
        return None
    strikes = df["strike"].values
    call_oi = df["call_oi"].values
    put_oi = df["put_oi"].values
    pains = []
    for Sc in strikes:
        call_loss = np.sum(call_oi * np.maximum(0, Sc - strikes))
        put_loss = np.sum(put_oi * np.maximum(0, strikes - Sc))
        pains.append(call_loss + put_loss)
    idx = int(np.argmin(pains))
    return float(strikes[idx])


# ─── Data fetchers (Yahoo Finance) ────────────────────────────────────────────
@st.cache_data(ttl=55, show_spinner=False)
def fetch_spot_price(ticker):
    if yf is None:
        return None
    try:
        t = yf.Ticker(ticker)
        h = t.history(period="5d", interval="1d")
        if h is None or h.empty:
            return None
        return float(h["Close"].iloc[-1])
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_expiries(ticker):
    if yf is None:
        return []
    try:
        t = yf.Ticker(ticker)
        exps = t.options
        return list(exps) if exps else []
    except Exception:
        return []


@st.cache_data(ttl=55, show_spinner=False)
def fetch_option_chain_raw(ticker, expiry):
    if yf is None:
        return None, None
    try:
        t = yf.Ticker(ticker)
        oc = t.option_chain(expiry)
        return oc.calls, oc.puts
    except Exception:
        return None, None


def build_demo_chain(spot):
    """Synthetic chain used only when live data is unavailable (off-hours,
    network hiccup, or Yahoo rate-limit) so the layout stays inspectable."""
    step = max(round(spot * 0.01, 0), 1)
    strikes = np.round(np.arange(spot - 20 * step, spot + 20 * step, step), 2)
    rows = []
    for K in strikes:
        dist = abs(K - spot) / max(spot, 1)
        base_iv = 18 + dist * 60
        call_oi = max(int(5000 * math.exp(-((K - spot - step * 3) ** 2) / (2 * (spot * 0.03) ** 2))), 10)
        put_oi = max(int(5000 * math.exp(-((K - spot + step * 3) ** 2) / (2 * (spot * 0.03) ** 2))), 10)
        rows.append({
            "strike": float(K),
            "call_oi": call_oi, "call_iv": base_iv + np.random.uniform(-1, 1),
            "call_ltp": max(spot - K, 0.5) + np.random.uniform(0.1, 2),
            "call_volume": int(call_oi * 0.3),
            "put_oi": put_oi, "put_iv": base_iv + np.random.uniform(-1, 1),
            "put_ltp": max(K - spot, 0.5) + np.random.uniform(0.1, 2),
            "put_volume": int(put_oi * 0.3),
        })
    return pd.DataFrame(rows)


def get_option_chain(ticker, expiry, spot, T):
    """Returns (df, data_source) where df has strike/call_*/put_* incl. greeks."""
    calls, puts = fetch_option_chain_raw(ticker, expiry)
    if calls is None or puts is None or calls.empty or puts.empty or spot is None:
        demo = build_demo_chain(spot if spot else 100.0)
        df = demo
        source = "DEMO"
    else:
        c = calls[["strike", "openInterest", "impliedVolatility", "lastPrice", "volume"]].copy()
        c.columns = ["strike", "call_oi", "call_iv_raw", "call_ltp", "call_volume"]
        p = puts[["strike", "openInterest", "impliedVolatility", "lastPrice", "volume"]].copy()
        p.columns = ["strike", "put_oi", "put_iv_raw", "put_ltp", "put_volume"]
        df = pd.merge(c, p, on="strike", how="outer").sort_values("strike").reset_index(drop=True)
        for col in ["call_oi", "call_volume", "put_oi", "put_volume"]:
            df[col] = df[col].fillna(0).astype(float)
        for col in ["call_ltp", "put_ltp", "call_iv_raw", "put_iv_raw"]:
            df[col] = df[col].fillna(0.0)
        # yfinance IV is a fraction (e.g. 0.18) -> convert to % ; fall back to
        # solving IV from last traded price when Yahoo's field is 0/missing.
        call_iv_pct, put_iv_pct = [], []
        for _, row in df.iterrows():
            civ = row["call_iv_raw"] * 100
            if civ <= 0.5 and row["call_ltp"] > 0:
                sv = _solve_iv(row["call_ltp"], spot, row["strike"], T, RISK_FREE_RATE, "CE")
                civ = sv * 100 if not np.isnan(sv) else 0.0
            call_iv_pct.append(civ)
            piv = row["put_iv_raw"] * 100
            if piv <= 0.5 and row["put_ltp"] > 0:
                sv = _solve_iv(row["put_ltp"], spot, row["strike"], T, RISK_FREE_RATE, "PE")
                piv = sv * 100 if not np.isnan(sv) else 0.0
            put_iv_pct.append(piv)
        df["call_iv"] = call_iv_pct
        df["put_iv"] = put_iv_pct
        df = df.drop(columns=["call_iv_raw", "put_iv_raw"])
        source = "LIVE"

    # Greeks per strike from Black-Scholes using the (solved / Yahoo) IV
    cd, cg, ct, cv, pd_, pg, pt, pv = [], [], [], [], [], [], [], []
    for _, row in df.iterrows():
        d, g, th, ve = _bs_greeks(spot, row["strike"], T, RISK_FREE_RATE, max(row["call_iv"], 0.01) / 100, "CE")
        cd.append(d); cg.append(g); ct.append(th); cv.append(ve)
        d, g, th, ve = _bs_greeks(spot, row["strike"], T, RISK_FREE_RATE, max(row["put_iv"], 0.01) / 100, "PE")
        pd_.append(d); pg.append(g); pt.append(th); pv.append(ve)
    df["call_delta"], df["call_gamma"], df["call_theta"], df["call_vega"] = cd, cg, ct, cv
    df["put_delta"], df["put_gamma"], df["put_theta"], df["put_vega"] = pd_, pg, pt, pv
    return df, source


# ─── Metrics + bias engine ────────────────────────────────────────────────────
def compute_metrics(df, spot):
    call_oi_total = float(df["call_oi"].sum())
    put_oi_total = float(df["put_oi"].sum())
    pcr = round(put_oi_total / call_oi_total, 3) if call_oi_total > 0 else 0.0
    atm = float(df.iloc[(df["strike"] - spot).abs().argsort()[:1]]["strike"].values[0])
    max_pain = compute_max_pain(df)
    total_gex, gex_series, gamma_flip_strike = compute_true_gex(df, spot)
    iv_rank, iv_pct = compute_iv_rank(df, atm)
    call_wall = float(df.loc[df["call_oi"].idxmax(), "strike"]) if call_oi_total > 0 else atm
    put_wall = float(df.loc[df["put_oi"].idxmax(), "strike"]) if put_oi_total > 0 else atm
    wall_width = abs(call_wall - put_wall)
    row_atm = df[df["strike"] == atm]
    atm_iv = float((row_atm["call_iv"].values[0] + row_atm["put_iv"].values[0]) / 2) if not row_atm.empty else 0.0
    return {
        "call_oi_total": call_oi_total, "put_oi_total": put_oi_total, "pcr": pcr,
        "atm": atm, "atm_iv": atm_iv, "max_pain": max_pain,
        "total_gex": total_gex, "gex_series": gex_series, "gamma_flip_strike": gamma_flip_strike,
        "iv_rank": iv_rank, "iv_pct": iv_pct, "call_wall": call_wall, "put_wall": put_wall,
        "wall_width": wall_width,
    }


def compute_bias_score(m, spot, prev_spot, gamma_flip_price, momentum):
    """Combines PCR skew, GEX sign, max-pain gravity, gamma-flip position, and
    short-term price momentum into a -100..+100 score. A simplified analogue
    of the NIFTY dashboard's multi-factor bias engine."""
    score = 0.0
    # PCR skew: PCR > 1.15 -> put-heavy (often contrarian bullish); < 0.85 -> call-heavy
    if m["pcr"] > 1.15:
        score += min((m["pcr"] - 1.0) * 60, 25)
    elif m["pcr"] < 0.85:
        score -= min((1.0 - m["pcr"]) * 60, 25)
    # GEX sign: positive GEX -> dealers long gamma -> pins/dampens -> mild pull to neutral
    if m["total_gex"] < 0:
        score += 15 if spot >= m["atm"] else -15
    # Max pain gravity: price tends to drift toward max pain into expiry
    if m["max_pain"]:
        diff_pct = (spot - m["max_pain"]) / max(spot, 1) * 100
        score -= max(min(diff_pct * 4, 20), -20)
    # Gamma flip position
    gf = gamma_flip_price if gamma_flip_price is not None else m.get("gamma_flip_strike")
    if gf:
        score += 15 if spot > gf else -15
    # Short-term momentum
    if prev_spot:
        chg_pct = (spot - prev_spot) / max(prev_spot, 1) * 100
        score += max(min(chg_pct * 20, 20), -20)
    score = max(min(score, 100), -100)
    if score >= 35:
        label, color = "BULLISH", "#16A34A"
    elif score <= -35:
        label, color = "BEARISH", "#DC2626"
    else:
        label, color = "NEUTRAL / RANGE", "#D97706"
    confidence = min(100, int(abs(score) * 1.2 + 20))
    return round(score, 1), label, color, confidence


def strategy_recommendation(bias_label, regime, iv_rank):
    ideas = []
    high_iv = iv_rank >= 60
    low_iv = iv_rank <= 35
    if bias_label == "BULLISH":
        ideas.append(("Bull Put Spread", "Sell OTM put spread below spot — collects premium with defined risk in an up-bias tape."))
        if low_iv:
            ideas.append(("Long Call / Call Debit Spread", "Low IV favours buying premium for directional upside exposure."))
    elif bias_label == "BEARISH":
        ideas.append(("Bear Call Spread", "Sell OTM call spread above spot — collects premium with defined risk in a down-bias tape."))
        if low_iv:
            ideas.append(("Long Put / Put Debit Spread", "Low IV favours buying premium for directional downside exposure."))
    else:
        if "PINNED" in regime or "RANGE" in regime:
            ideas.append(("Iron Condor / Short Strangle", "Positive GEX + range regime favours premium selling around max pain."))
        if high_iv:
            ideas.append(("Iron Butterfly at Max Pain", "Elevated IV rank + range-bound regime — sell the wings near max pain."))
    if "TREND / EXPANSION" in regime:
        ideas.append(("Long Straddle / Strangle", "Negative GEX + expansion regime — volatility likely to widen further."))
    if not ideas:
        ideas.append(("Stand Aside", "No high-conviction setup — signals are mixed or too weak to size a trade."))
    return ideas


def compute_oi_momentum(df, prev_df):
    """Per-strike OI-change 'buyer/seller' matrix. Classic OI interpretation:
    price up + OI up = long buildup; price up + OI down = short covering;
    price down + OI up = short buildup; price down + OI down = long unwinding.
    Approximated here using LTP change and OI change strike-by-strike between
    this refresh and the previous one (session-local, resets each session)."""
    if prev_df is None or prev_df.empty:
        return None
    merged = pd.merge(df[["strike", "call_oi", "call_ltp", "put_oi", "put_ltp"]],
                       prev_df[["strike", "call_oi", "call_ltp", "put_oi", "put_ltp"]],
                       on="strike", how="inner", suffixes=("", "_prev"))
    if merged.empty:
        return None

    def _tag(oi_chg, px_chg):
        if oi_chg > 0 and px_chg >= 0:
            return "Long Buildup"
        if oi_chg > 0 and px_chg < 0:
            return "Short Buildup"
        if oi_chg <= 0 and px_chg >= 0:
            return "Short Covering"
        return "Long Unwinding"

    merged["call_oi_chg"] = merged["call_oi"] - merged["call_oi_prev"]
    merged["call_px_chg"] = merged["call_ltp"] - merged["call_ltp_prev"]
    merged["call_verdict"] = [
        _tag(o, p) for o, p in zip(merged["call_oi_chg"], merged["call_px_chg"])
    ]
    merged["put_oi_chg"] = merged["put_oi"] - merged["put_oi_prev"]
    merged["put_px_chg"] = merged["put_ltp"] - merged["put_ltp_prev"]
    merged["put_verdict"] = [
        _tag(o, p) for o, p in zip(merged["put_oi_chg"], merged["put_px_chg"])
    ]
    return merged[["strike", "call_oi_chg", "call_px_chg", "call_verdict",
                    "put_oi_chg", "put_px_chg", "put_verdict"]]


# ─── UI helpers ────────────────────────────────────────────────────────────────
def _chip(label, value, color):
    st.markdown(
        f"""<div style="background:{color}18;border:1px solid {color}55;border-radius:10px;
        padding:10px 14px;text-align:center;">
        <div style="font-size:11px;color:#6B7280;font-weight:600;text-transform:uppercase;">{label}</div>
        <div style="font-size:20px;color:{color};font-weight:800;">{value}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def _data_status_banner(source, market_open):
    if source == "LIVE" and market_open:
        st.success(f"🟢 LIVE — {ASSET_CONFIG['ticker']} options chain via Yahoo Finance (exchange-delayed) · {et_str()}")
    elif source == "LIVE" and not market_open:
        st.info(f"🌙 MARKET CLOSED — showing last available {ASSET_CONFIG['ticker']} chain · {et_str()}")
    else:
        st.warning(f"🟡 DEMO MODE — live data unavailable (network hiccup, rate-limit, or off-hours) · {et_str()}")


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    refresh_secs = st.selectbox("Auto-refresh interval", [30, 60, 120, 300], index=1,
                                 format_func=lambda x: f"{x} sec")
    st.caption(ASSET_CONFIG["underlying_desc"])
    st.caption("Data: Yahoo Finance (yfinance) — free, exchange-delayed options chain.")

st_autorefresh(interval=refresh_secs * 1000, key="autorefresh")

# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    f"""<div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;">
    <span style="font-size:34px;">{ASSET_CONFIG['icon']}</span>
    <div>
      <div style="font-size:26px;font-weight:800;color:#1A1A2E;">Shantanu's Options Dashboard — {ASSET_CONFIG['short_name']}</div>
      <div style="font-size:13px;color:#6B7280;">{ASSET_CONFIG['underlying_desc']}</div>
    </div>
    </div>""",
    unsafe_allow_html=True,
)

market_open = is_market_hours()
spot = fetch_spot_price(TICKER)
if spot is None:
    spot = ASSET_CONFIG.get("demo_spot", 100.0)

expiries = fetch_expiries(TICKER)
if not expiries:
    today = now_et().date()
    expiries = [(today + timedelta(days=d)).isoformat() for d in (7, 14, 30) if True]

col_a, col_b = st.columns([2, 1])
with col_a:
    expiry = st.selectbox("Expiry", expiries, index=0)
with col_b:
    st.metric(f"{ASSET_CONFIG['ticker']} Spot", f"{ASSET_CONFIG['currency']}{spot:,.2f}")

try:
    exp_date = datetime.strptime(expiry, "%Y-%m-%d").date()
    T = max((exp_date - now_et().date()).days, 0) / 365.0
    if T <= 0:
        T = 1 / 365.0
except Exception:
    T = 7 / 365.0

df, source = get_option_chain(TICKER, expiry, spot, T)
_data_status_banner(source, market_open)

m = compute_metrics(df, spot)

# session-local snapshot history for momentum / temporal IV rank
if "snap_history" not in st.session_state:
    st.session_state.snap_history = []
if "prev_chain" not in st.session_state:
    st.session_state.prev_chain = None
if "prev_spot" not in st.session_state:
    st.session_state.prev_spot = None
if "verdict_log" not in st.session_state:
    st.session_state.verdict_log = []

iv_hist = _load_iv_history()
temporal_rank, is_temporal = compute_temporal_iv_rank(m["atm_iv"], iv_hist)
iv_hist.append({"ts": now_et().isoformat(), "atm_iv": m["atm_iv"]})
_save_iv_history(iv_hist)

gamma_flip_price = compute_gamma_flip_true(df, spot, T) or m["gamma_flip_strike"]
momentum = (spot - st.session_state.prev_spot) if st.session_state.prev_spot else 0.0
regime, vol_regime, near_flip = classify_gamma_regime(
    m["total_gex"], m["wall_width"], momentum,
    temporal_rank if is_temporal else m["iv_rank"], spot, gamma_flip_price
)
bias_score, bias_label, bias_color, confidence = compute_bias_score(
    m, spot, st.session_state.prev_spot, gamma_flip_price, momentum
)

# ── SECTION 1: MARKET SENTIMENT ───────────────────────────────────────────────
st.markdown("#### 📍 Section 1 — Market Snapshot")
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1: _chip("PCR (OI)", f"{m['pcr']:.2f}", "#2563EB")
with c2: _chip("Max Pain", f"{m['max_pain']:.1f}" if m['max_pain'] else "—", "#7C3AED")
with c3: _chip("ATM IV", f"{m['atm_iv']:.1f}%", "#0891B2")
with c4: _chip("Net GEX", f"{m['total_gex']/1e6:,.1f}M", "#059669" if m['total_gex'] >= 0 else "#DC2626")
with c5: _chip("Gamma Flip", f"{gamma_flip_price:.1f}" if gamma_flip_price else "—", "#D97706")
with c6: _chip("Regime", regime, "#4338CA")

st.write("")

# ── SECTION 2: BIAS ENGINE + STRATEGY ─────────────────────────────────────────
st.markdown("#### 🎯 Section 2 — Bias Engine + Strategy")
bcol1, bcol2 = st.columns([1, 2])
with bcol1:
    st.markdown(
        f"""<div style="background:{bias_color}15;border:2px solid {bias_color};border-radius:14px;
        padding:18px;text-align:center;">
        <div style="font-size:12px;color:#6B7280;font-weight:700;text-transform:uppercase;">Bias Score</div>
        <div style="font-size:40px;font-weight:900;color:{bias_color};">{bias_score:+.0f}</div>
        <div style="font-size:16px;font-weight:800;color:{bias_color};">{bias_label}</div>
        <div style="font-size:12px;color:#6B7280;">Confidence: {confidence}%</div>
        </div>""",
        unsafe_allow_html=True,
    )
with bcol2:
    ideas = strategy_recommendation(bias_label, regime, temporal_rank if is_temporal else m['iv_rank'])
    for name, why in ideas:
        st.markdown(f"**{name}** — {why}")

st.write("")

# ── SECTION 3: KEY PRICE LEVELS ────────────────────────────────────────────────
st.markdown("#### 📐 Section 3 — Key Price Levels")
l1, l2, l3, l4, l5 = st.columns(5)
with l1: _chip("Call Wall", f"{m['call_wall']:.1f}", "#DC2626")
with l2: _chip("Put Wall", f"{m['put_wall']:.1f}", "#16A34A")
with l3: _chip("ATM Strike", f"{m['atm']:.1f}", "#374151")
with l4: _chip("Spot", f"{spot:,.2f}", "#111827")
with l5: _chip("IV Rank (Temporal)", f"{temporal_rank:.0f}%" if is_temporal else f"{m['iv_rank']:.0f}%*", "#0891B2")
if not is_temporal:
    st.caption("* Cross-sectional smile-position rank shown — temporal rank needs more history (accrues across sessions).")

st.write("")

# ── SECTION 4: STRIKE-WISE CHARTS ─────────────────────────────────────────────
st.markdown("#### 📊 Section 4 — Strike-wise Structure")
band = df[(df["strike"] >= m["atm"] - 12 * max((df["strike"].diff().median() or 1), 1)) &
          (df["strike"] <= m["atm"] + 12 * max((df["strike"].diff().median() or 1), 1))]
if band.empty:
    band = df

r1c1, r1c2 = st.columns(2)
with r1c1:
    fig = go.Figure()
    fig.add_bar(x=band["strike"], y=band["call_oi"], name="Call OI", marker_color="#DC2626")
    fig.add_bar(x=band["strike"], y=band["put_oi"], name="Put OI", marker_color="#16A34A")
    fig.update_layout(title="Open Interest by Strike", barmode="group", height=340,
                       margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)
with r1c2:
    gex_band = m["gex_series"].reindex(band["strike"]).fillna(0)
    fig = go.Figure()
    fig.add_bar(x=gex_band.index, y=gex_band.values,
                marker_color=["#059669" if v >= 0 else "#DC2626" for v in gex_band.values])
    fig.update_layout(title="Net GEX by Strike", height=340, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

r2c1, r2c2 = st.columns(2)
with r2c1:
    fig = go.Figure()
    fig.add_scatter(x=band["strike"], y=band["call_iv"], name="Call IV", mode="lines+markers", line=dict(color="#DC2626"))
    fig.add_scatter(x=band["strike"], y=band["put_iv"], name="Put IV", mode="lines+markers", line=dict(color="#16A34A"))
    fig.update_layout(title="IV Smile", height=340, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)
with r2c2:
    fig = go.Figure()
    fig.add_bar(x=band["strike"], y=band["call_gamma"], name="Call Gamma", marker_color="#DC2626")
    fig.add_bar(x=band["strike"], y=band["put_gamma"], name="Put Gamma", marker_color="#16A34A")
    fig.update_layout(title="Gamma by Strike", barmode="group", height=340, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.write("")

# ── SECTION 5: OI MOMENTUM / BUYER-SELLER MATRIX ──────────────────────────────
st.markdown("#### 🔁 Section 5 — OI Momentum Matrix (session-local)")
momentum_df = compute_oi_momentum(df, st.session_state.prev_chain)
if momentum_df is None:
    st.info("Building baseline — the momentum matrix populates from the second refresh onward this session.")
else:
    show = momentum_df.reindex(momentum_df["strike"].sub(m["atm"]).abs().sort_values().index).head(16).sort_values("strike")
    st.dataframe(
        show.rename(columns={
            "strike": "Strike", "call_oi_chg": "Call ΔOI", "call_px_chg": "Call ΔLTP",
            "call_verdict": "Call Verdict", "put_oi_chg": "Put ΔOI", "put_px_chg": "Put ΔLTP",
            "put_verdict": "Put Verdict",
        }),
        use_container_width=True, hide_index=True,
    )
    verdict_summary = f"{now_et().strftime('%H:%M:%S')} — Bias {bias_label} ({bias_score:+.0f}) · Regime {regime} · GEX {'+' if m['total_gex']>=0 else ''}{m['total_gex']/1e6:.1f}M"
    st.session_state.verdict_log.append(verdict_summary)
    st.session_state.verdict_log = st.session_state.verdict_log[-20:]
    with st.expander("Verdict log (this session)"):
        for line in reversed(st.session_state.verdict_log):
            st.text(line)

st.session_state.prev_chain = df.copy()
st.session_state.prev_spot = spot

st.write("")

# ── SECTION 6: RAW OPTION CHAIN TABLE ─────────────────────────────────────────
st.markdown("#### 📋 Section 6 — Raw Option Chain")
raw_cols = ["call_oi", "call_iv", "call_ltp", "call_delta", "call_gamma", "strike",
            "put_delta", "put_gamma", "put_ltp", "put_iv", "put_oi"]
show_raw = band[raw_cols].rename(columns={
    "call_oi": "Call OI", "call_iv": "Call IV%", "call_ltp": "Call LTP",
    "call_delta": "Call Δ", "call_gamma": "Call Γ", "strike": "Strike",
    "put_delta": "Put Δ", "put_gamma": "Put Γ", "put_ltp": "Put LTP",
    "put_iv": "Put IV%", "put_oi": "Put OI",
})
st.dataframe(show_raw, use_container_width=True, hide_index=True, height=420)

st.markdown("---")
st.caption(
    f"⚠️ Educational / paper-trading tool only — not investment advice. "
    f"{ASSET_CONFIG['ticker']} options data is exchange-delayed and is a proxy for {ASSET_CONFIG['short_name']}; "
    f"it does not represent live CME futures option pricing. Open interest updates once per session, not tick-by-tick. "
    f"No trades are placed by this app."
)
