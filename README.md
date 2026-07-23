# S&P 500 / SPY Options Dashboard

Streamlit clone of the NIFTY options-analysis dashboard, adapted for **ES1! (E-mini S&P 500 futures)** using free live data.

## Why SPY and not the future itself
There is no free, live options-chain (OI/Greeks) API for CME futures (ES/NQ/GC/CL/6E and their micros) —
that data is sold commercially. **SPY** is the most liquid, listed-options, free-data-accessible
ETF proxy for this underlying, sourced via Yahoo Finance (`yfinance`).

## What's inside
- Live spot + options chain fetch (Yahoo Finance, exchange-delayed ~15 min, OI updates once/session)
- Black-Scholes Greeks (delta/gamma/theta/vega) + IV solver (matches the original engine's formulas)
- Net GEX, gamma flip (price-domain), max pain, PCR, IV rank (cross-sectional + session-accruing temporal)
- Bias score (-100..+100), regime classifier, strategy suggestions
- Session-local OI momentum ("buyer/seller") matrix + verdict log
- Strike-wise OI / GEX / IV-smile / Gamma charts, raw option chain table
- Auto-refresh (30/60/120/300s, selectable), demo-mode fallback if live data is unavailable

## Deploy to Streamlit Community Cloud
1. Create a new GitHub repo and push the contents of this folder (`app.py`, `requirements.txt`, this README) to it — one file each at the repo root.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → pick the repo/branch.
3. Set **Main file path** to `app.py`. Deploy.
4. Python 3.11 is recommended (Streamlit Cloud's default) — no extra secrets or API keys are required, since Yahoo Finance access via `yfinance` needs none.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Known limitations
- **Proxy, not the future**: SPY price/IV will diverge from ES1! (E-mini S&P 500 futures) pricing (different underlying, dividend/carry effects, contract specs).
- **Delayed data**: Yahoo Finance quotes/chains are exchange-delayed, not real-time tick data.
- **OI cadence**: Open interest is an end-of-day figure that Yahoo refreshes once per session, not intraday — the "OI momentum matrix" is session-local (compares refreshes within your browser session, not true intraday OI ticks).
- **Ephemeral history**: the IV-history JSON file used for temporal IV rank resets whenever the Streamlit Cloud container restarts/redeploys.
- **Not investment advice**: educational / paper-trading use only. No trades are placed by this app.
