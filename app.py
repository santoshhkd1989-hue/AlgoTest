import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="SMC Pro Swing Scanner & Backtester (NSE/BSE + Dhan)",
    layout="wide",
    page_icon="🏦"
)

# ==========================================
# 1. NSE & BSE UNIVERSE FOR SWING SCANNING
# ==========================================

NSE_BSE_UNIVERSE = {
    "NSE High-Conviction Swing Basket": [
        "ONGC.NS", "SUZLON.NS", "CUPID.NS", "VPRPL.NS", "RELIANCE.NS",
        "SBIN.NS", "TATAMOTORS.NS", "HAL.NS", "BEL.NS", "TRENT.NS",
        "BAJFINANCE.NS", "LT.NS", "NTPC.NS", "POWERGRID.NS", "COALINDIA.NS",
        "TATASTEEL.NS", "HINDALCO.NS", "SUNPHARMA.NS", "INFY.NS", "TCS.NS"
    ],
    "BSE Momentum & Midcap Basket": [
        "ONGC.BO", "SUZLON.BO", "RELIANCE.BO", "SBIN.BO", "TATAMOTORS.BO",
        "BEL.BO", "HAL.BO", "BHEL.BO", "IRFC.BO", "RVNL.BO",
        "PFC.BO", "RECLTD.BO", "TATASTEEL.BO", "LT.BO", "NTPC.BO"
    ]
}

# ==========================================
# 2. SMART MONEY CONCEPTS (SMC) ENGINE
# ==========================================

def calculate_smc_features(df: pd.DataFrame, swing_len: int = 5) -> pd.DataFrame:
    """
    Detects institutional Smart Money Concepts (SMC) on Daily OHLCV data:
    - Swing Highs & Swing Lows (Market Structure)
    - Break of Structure (BOS) & Change of Character (CHoCH)
    - Bullish & Bearish Fair Value Gaps (FVG / Imbalance)
    - Bullish Order Block (Last down candle before impulsive displacement + BOS)
    - Liquidity Sweep (Stop hunt below recent swing low followed by strong rejection close)
    - Premium / Discount Zone (50% Equilibrium of 40-day Dealing Range)
    - 9/28 EMA, RSI(14), ATR(14) for confluence and SL/Target sizing
    """
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA28"] = df["Close"].ewm(span=28, adjust=False).mean()

    # RSI (14)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = (100 - (100 / (1 + rs))).fillna(50)

    # ATR (14)
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean().bfill()

    # Rolling Swing High & Swing Low (Structure Levels)
    df["Swing_High"] = df["High"].shift(1).rolling(swing_len * 2, min_periods=3).max()
    df["Swing_Low"] = df["Low"].shift(1).rolling(swing_len * 2, min_periods=3).min()

    # Dealing Range (40-day Range) -> Discount (< 50%) vs Premium (> 50%)
    df["Range_High"] = df["High"].rolling(40, min_periods=10).max()
    df["Range_Low"] = df["Low"].rolling(40, min_periods=10).min()
    df["Equilibrium_50"] = (df["Range_High"] + df["Range_Low"]) / 2.0
    df["In_Discount"] = df["Close"] <= df["Equilibrium_50"]

    # Bullish Fair Value Gap (FVG): Low[i] > High[i-2] with impulsive middle candle
    df["Bullish_FVG"] = df["Low"] > df["High"].shift(2)
    df["FVG_Top"] = np.where(df["Bullish_FVG"], df["Low"], np.nan)
    df["FVG_Bottom"] = np.where(df["Bullish_FVG"], df["High"].shift(2), np.nan)
    df["Active_FVG_Top"] = pd.Series(df["FVG_Top"], index=df.index).ffill()
    df["Active_FVG_Bottom"] = pd.Series(df["FVG_Bottom"], index=df.index).ffill()

    # Bullish Break of Structure (BOS) & Change of Character (CHoCH)
    prev_downtrend = df["EMA9"].shift(3) < df["EMA28"].shift(3)
    struct_break_up = (df["Close"] > df["Swing_High"]) & (df["Close"].shift(1) <= df["Swing_High"].shift(1))
    df["Bull_CHoCH"] = struct_break_up & prev_downtrend
    df["Bull_BOS"] = struct_break_up & (~prev_downtrend)

    # Liquidity Sweep (Sell-Side Liquidity Grab): Low pierces Swing_Low, but Close reclaims above Swing_Low
    df["Liq_Sweep_Bull"] = (df["Low"] < df["Swing_Low"]) & (df["Close"] > df["Swing_Low"])

    # Bullish Order Block (OB): Last bearish candle prior to impulsive move (> 1.2x ATR)
    is_bearish = df["Close"] < df["Open"]
    impulsive_next = (df["Close"].shift(-1) - df["Open"].shift(-1)) > (1.1 * df["ATR"])
    df["Is_Bull_OB"] = is_bearish & impulsive_next
    df["OB_High"] = np.where(df["Is_Bull_OB"], df["High"], np.nan)
    df["OB_Low"] = np.where(df["Is_Bull_OB"], df["Low"], np.nan)
    df["Active_OB_High"] = pd.Series(df["OB_High"], index=df.index).ffill()
    df["Active_OB_Low"] = pd.Series(df["OB_Low"], index=df.index).ffill()

    # Retest / Mitigation of Bullish Order Block or FVG
    df["OB_Mitigated"] = (df["Low"] <= df["Active_OB_High"] * 1.01) & (df["Close"] >= df["Active_OB_Low"])
    df["FVG_Mitigated"] = (df["Low"] <= df["Active_FVG_Top"] * 1.01) & (df["Close"] >= df["Active_FVG_Bottom"])

    return df

def score_stock_smc(symbol: str, df: pd.DataFrame) -> Optional[dict]:
    if df is None or len(df) < 40:
        return None
    df = calculate_smc_features(df)
    last = df.iloc[-1]
    recent5 = df.iloc[-5:]

    close_p = float(last["Close"])
    atr = float(last["ATR"])
    rsi = float(last["RSI"])

    # Check SMC Triggers in the last 5 trading days
    has_bos = bool(recent5["Bull_BOS"].any())
    has_choch = bool(recent5["Bull_CHoCH"].any())
    has_sweep = bool(recent5["Liq_Sweep_Bull"].any())
    has_fvg = bool(recent5["Bullish_FVG"].any() or last["FVG_Mitigated"])
    has_ob = bool(last["OB_Mitigated"])
    ema_bull = bool(last["EMA9"] > last["EMA28"])
    in_discount = bool(last["In_Discount"])

    # Institutional SMC Confluence Score (out of 100)
    score = 0
    tags = []
    if has_choch:
        score += 30
        tags.append("Bull CHoCH")
    elif has_bos:
        score += 25
        tags.append("Bull BOS")

    if has_ob:
        score += 25
        tags.append("OB Retest")
    if has_sweep:
        score += 25
        tags.append("SSL Liq Sweep")
    if has_fvg:
        score += 15
        tags.append("Bull FVG")
    if ema_bull and 48 <= rsi <= 72:
        score += 15
        tags.append("EMA 9>28 + RSI")
    if in_discount:
        score += 10
        tags.append("Discount Zone")

    score = min(100, score)

    # Professional SMC Entry, Stop-Loss (Below OB Low / Structure Low - 0.5 ATR), and Targets (Swing High Liquidity Pool)
    ob_low = float(last["Active_OB_Low"]) if not np.isnan(last["Active_OB_Low"]) else (close_p - 1.5 * atr)
    struct_low = float(last["Swing_Low"]) if not np.isnan(last["Swing_Low"]) else (close_p - 1.5 * atr)
    sl_candidate = min(ob_low, struct_low) - (0.3 * atr)
    # Keep SL reasonable (between 2% and 7% below entry)
    max_sl_dist = close_p * 0.065
    min_sl_dist = close_p * 0.02
    risk_per_share = np.clip(close_p - sl_candidate, min_sl_dist, max_sl_dist)
    sl_price = round(close_p - risk_per_share, 2)

    target_1 = round(close_p + (risk_per_share * 2.0), 2)   # 1:2 R:R
    target_2 = round(close_p + (risk_per_share * 3.2), 2)   # 1:3.2 R:R (Buy-Side Liquidity Pool)
    rr_ratio = round((target_2 - close_p) / risk_per_share, 2) if risk_per_share > 0 else 0.0

    exch = "BSE" if symbol.endswith(".BO") else "NSE"
    clean_sym = symbol.replace(".NS", "").replace(".BO", "")

    return {
        "Symbol": clean_sym,
        "Yahoo_Ticker": symbol,
        "Exchange": exch,
        "CMP (₹)": round(close_p, 2),
        "SMC Score": score,
        "SMC Setup": " + ".join(tags) if tags else "Neutral Structure",
        "Zone": "DISCOUNT (Buy Zone)" if in_discount else "PREMIUM (Momentum)",
        "Entry (₹)": round(close_p, 2),
        "Stop Loss (₹)": sl_price,
        "Target 1 (1:2) (₹)": target_1,
        "Target 2 (BSL) (₹)": target_2,
        "R:R": f"1:{rr_ratio}",
        "RSI (14)": round(rsi, 1),
        "Risk/Share (₹)": round(risk_per_share, 2)
    }

@st.cache_data(ttl=900)
def fetch_and_scan_universe(tickers: list[str]) -> tuple[pd.DataFrame, dict]:
    results = []
    raw_histories = {}
    data = yf.download(tickers, period="6mo", interval="1d", group_by="ticker", progress=False, threads=True)
    for t in tickers:
        try:
            sub_df = data[t].dropna(how="all") if len(tickers) > 1 else data.dropna(how="all")
            if not sub_df.empty and len(sub_df) >= 40:
                raw_histories[t] = sub_df
                scored = score_stock_smc(t, sub_df)
                if scored:
                    results.append(scored)
        except Exception:
            continue
    out_df = pd.DataFrame(results)
    if not out_df.empty:
        out_df = out_df.sort_values(by=["SMC Score", "RSI (14)"], ascending=[False, False])
    return out_df, raw_histories

# ==========================================
# 3. SMC SWING BACKTESTER ENGINE
# ==========================================

def backtest_smc_swing(df: pd.DataFrame, capital: float = 100000.0, risk_pct: float = 1.5, rr_mult: float = 2.5) -> tuple[pd.DataFrame, dict]:
    """
    Simulates institutional SMC Swing Entries & Exits on historical daily bars:
    - Entry: Bullish BOS/CHoCH or Liquidity Sweep or OB Mitigation + EMA9 > EMA28
    - Exit: Stop-Loss below Order Block / Swing Low OR Target at R:R multiple OR Bearish CHoCH (Close < Swing Low)
    """
    df = calculate_smc_features(df)
    in_trade = False
    entry_p, sl_p, tgt_p, qty = 0.0, 0.0, 0.0, 0
    entry_date, setup_reason = None, ""
    trades = []
    equity = capital

    for i in range(40, len(df)):
        row = df.iloc[i]
        dt = df.index[i]
        high, low, close = float(row["High"]), float(row["Low"]), float(row["Close"])

        if not in_trade:
            smc_entry = (
                bool(row["Bull_CHoCH"]) or
                bool(row["Bull_BOS"]) or
                (bool(row["Liq_Sweep_Bull"]) and close > float(row["EMA9"])) or
                (bool(row["OB_Mitigated"]) and float(row["EMA9"]) > float(row["EMA28"]))
            )
            if smc_entry:
                in_trade = True
                entry_p = round(close, 2)
                entry_date = dt.strftime("%Y-%m-%d")
                atr = float(row["ATR"])
                risk_per_sh = max(entry_p * 0.02, min(entry_p * 0.06, 1.5 * atr))
                sl_p = round(entry_p - risk_per_sh, 2)
                tgt_p = round(entry_p + (risk_per_sh * rr_mult), 2)
                risk_rupees = equity * (risk_pct / 100.0)
                qty = max(1, int(risk_rupees / risk_per_sh))
                if row["Bull_CHoCH"]:
                    setup_reason = "Bullish CHoCH"
                elif row["Liq_Sweep_Bull"]:
                    setup_reason = "SSL Liquidity Sweep"
                elif row["Bull_BOS"]:
                    setup_reason = "Bullish BOS"
                else:
                    setup_reason = "Order Block Mitigation"
        else:
            # Check SL, Target, or Bearish Structure Break (Close < Swing_Low)
            exit_reason = None
            exit_p = close
            if low <= sl_p:
                exit_reason = "SMC Stop-Loss (OB Invalidated)"
                exit_p = sl_p
            elif high >= tgt_p:
                exit_reason = f"Liquidity Pool Target (1:{rr_mult})"
                exit_p = tgt_p
            elif close < float(row["Swing_Low"]) and (dt - pd.to_datetime(entry_date)).days >= 3:
                exit_reason = "Bearish CHoCH Structure Exit"
                exit_p = close

            if exit_reason:
                pnl = round((exit_p - entry_p) * qty, 2)
                equity += pnl
                trades.append({
                    "Entry Date": entry_date,
                    "Exit Date": dt.strftime("%Y-%m-%d"),
                    "SMC Entry Trigger": setup_reason,
                    "Qty": qty,
                    "Entry (₹)": entry_p,
                    "Stop Loss (₹)": sl_p,
                    "Exit (₹)": round(exit_p, 2),
                    "Exit Reason": exit_reason,
                    "Net PnL (₹)": pnl,
                    "Equity (₹)": round(equity, 2)
                })
                in_trade = False

    t_df = pd.DataFrame(trades)
    if t_df.empty:
        return t_df, {"Net Profit": 0.0, "Win Rate": 0.0, "Total Trades": 0, "Final Equity": capital}

    wins = t_df[t_df["Net PnL (₹)"] > 0]
    win_rate = (len(wins) / len(t_df)) * 100.0
    return t_df, {
        "Net Profit": round(equity - capital, 2),
        "Win Rate": round(win_rate, 1),
        "Total Trades": len(t_df),
        "Final Equity": round(equity, 2)
    }

# ==========================================
# 4. DHANHQ V2 CNC DELIVERY API CLIENT
# ==========================================

class DhanSwingBroker:
    BASE_URL = "https://api.dhan.co/v2"

    def __init__(self, client_id: str, access_token: str):
        self.client_id = client_id.strip()
        self.access_token = access_token.strip()
        self.headers = {
            "access-token": self.access_token,
            "client-id": self.client_id,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

    def place_cnc_swing_order(
        self,
        security_id: str,
        exchange_segment: str,
        transaction_type: str,
        quantity: int,
        order_type: str = "MARKET",
        price: float = 0.0
    ) -> dict:
        payload = {
            "dhanClientId": self.client_id,
            "transactionType": transaction_type,
            "exchangeSegment": exchange_segment,  # 'NSE_EQ' or 'BSE_EQ'
            "productType": "CNC",                 # Cash & Carry for Swing Delivery
            "orderType": order_type,
            "validity": "DAY",
            "securityId": str(security_id),
            "quantity": int(quantity),
            "disclosedQuantity": 0,
            "price": float(price),
            "triggerPrice": 0.0,
            "afterMarketOrder": False
        }
        resp = requests.post(f"{self.BASE_URL}/orders", json=payload, headers=self.headers, timeout=10)
        return resp.json()

@st.cache_data(ttl=3600)
def load_dhan_equity_master() -> pd.DataFrame:
    url = "https://images.dhan.co/api-data/api-scrip-master.csv"
    cols = ["SEM_EXM_EXCH_ID", "SEM_SEGMENT", "SEM_SMST_SECURITY_ID", "SEM_TRADING_SYMBOL", "SEM_INSTRUMENT_NAME"]
    try:
        df = pd.read_csv(url, usecols=cols, low_memory=False)
        return df[(df["SEM_SEGMENT"] == "E") & (df["SEM_INSTRUMENT_NAME"] == "EQUITY")]
    except Exception:
        return pd.DataFrame()

# ==========================================
# 5. MAIN DASHBOARD UI
# ==========================================

st.title("🏦 SMC Institutional Swing Trader — NSE & BSE Scanner + Dhan CNC Execution")
st.caption("Smart Money Concepts (BOS • CHoCH • Order Blocks • Fair Value Gaps • Liquidity Sweeps • Discount/Premium Zones)")

tabs = st.tabs([
    "🔍 1. NSE & BSE Live SMC Scanner",
    "📐 2. Interactive SMC Structure Chart",
    "📊 3. SMC Swing Strategy Backtester",
    "🟢 4. Dhan CNC Swing Order Terminal"
])

# ------------------------------------------
# TAB 1: NSE & BSE LIVE SMC MARKET SCANNER
# ------------------------------------------
with tabs[0]:
    st.subheader("🔍 Scan NSE & BSE Stocks for Institutional SMC Setups")
    c1, c2, c3, c4 = st.columns([1.8, 2.2, 1, 1])
    basket_choice = c1.selectbox("Select Market Universe", list(NSE_BSE_UNIVERSE.keys()))
    custom_tickers = c2.text_input(
        "Add Custom NSE (.NS) or BSE (.BO) Symbols",
        value="ONGC.NS, SUZLON.NS, CUPID.NS, VPRPL.NS"
    )
    min_score = c3.slider("Min SMC Score", 0, 80, 25, step=5)
    capital_size = c4.number_input("Swing Capital (₹)", min_value=10000, value=100000, step=10000)

    base_list = NSE_BSE_UNIVERSE[basket_choice]
    extra_list = [x.strip().upper() for x in custom_tickers.split(",") if x.strip()]
    full_tickers = list(dict.fromkeys(extra_list + base_list))

    if st.button("🚀 Scan NSE & BSE Market Now (SMC Engine)", type="primary", use_container_width=True):
        with st.spinner(f"Scanning {len(full_tickers)} NSE/BSE stocks for BOS, CHoCH, Order Blocks & FVGs..."):
            scan_df, raw_hist = fetch_and_scan_universe(full_tickers)
            st.session_state["scan_df"] = scan_df
            st.session_state["raw_hist"] = raw_hist

    if "scan_df" in st.session_state and not st.session_state["scan_df"].empty:
        df_filtered = st.session_state["scan_df"][st.session_state["scan_df"]["SMC Score"] >= min_score].copy()
        # Calculate Position Size based on 1.5% risk per trade
        risk_rupees = capital_size * 0.0
