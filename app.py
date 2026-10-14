import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(
    page_title="SMC Swing Trader — Full DhanHQ v2 Integration",
    layout="wide",
    page_icon="🏦"
)

# ==========================================
# 1. COMPLETE DHANHQ V2 API CLIENT
# ==========================================

class DhanHQClient:
    BASE_URL = "https://api.dhan.co/v2"

    def __init__(self, client_id: str, access_token: str):
        self.client_id = str(client_id).strip()
        self.access_token = str(access_token).strip()
        self.headers = {
            "access-token": self.access_token,
            "client-id": self.client_id,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

    def get_fund_limits(self) -> dict:
        try:
            r = requests.get(f"{self.BASE_URL}/fundlimit", headers=self.headers, timeout=10)
            return r.json() if r.status_code == 200 else {"error": r.text}
        except Exception as e:
            return {"error": str(e)}

    def get_holdings(self) -> list | dict:
        try:
            r = requests.get(f"{self.BASE_URL}/holdings", headers=self.headers, timeout=10)
            return r.json() if r.status_code == 200 else {"error": r.text}
        except Exception as e:
            return {"error": str(e)}

    def get_positions(self) -> list | dict:
        try:
            r = requests.get(f"{self.BASE_URL}/positions", headers=self.headers, timeout=10)
            return r.json() if r.status_code == 200 else {"error": r.text}
        except Exception as e:
            return {"error": str(e)}

    def get_orders(self) -> list | dict:
        try:
            r = requests.get(f"{self.BASE_URL}/orders", headers=self.headers, timeout=10)
            return r.json() if r.status_code == 200 else {"error": r.text}
        except Exception as e:
            return {"error": str(e)}

    def place_cnc_order(
        self,
        security_id: str,
        exchange_segment: str,
        transaction_type: str,
        quantity: int,
        order_type: str = "MARKET",
        price: float = 0.0,
        after_market: bool = False
    ) -> dict:
        payload = {
            "dhanClientId": self.client_id,
            "transactionType": transaction_type,
            "exchangeSegment": exchange_segment,  # 'NSE_EQ' or 'BSE_EQ'
            "productType": "CNC",
            "orderType": order_type,              # 'MARKET' or 'LIMIT'
            "validity": "DAY",
            "securityId": str(security_id),
            "quantity": int(quantity),
            "disclosedQuantity": 0,
            "price": float(price) if order_type == "LIMIT" else 0.0,
            "triggerPrice": 0.0,
            "afterMarketOrder": bool(after_market)
        }
        try:
            r = requests.post(f"{self.BASE_URL}/orders", json=payload, headers=self.headers, timeout=10)
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    def place_forever_oco_exit(
        self,
        security_id: str,
        exchange_segment: str,
        quantity: int,
        target_price: float,
        sl_price: float
    ) -> dict:
        """Places a Dhan Forever (GTT OCO) Sell Order for Swing Target + Stop Loss."""
        payload = {
            "dhanClientId": self.client_id,
            "orderFlag": "OCO",
            "transactionType": "SELL",
            "exchangeSegment": exchange_segment,
            "productType": "CNC",
            "orderType": "LIMIT",
            "validity": "DAY",
            "securityId": str(security_id),
            "quantity": int(quantity),
            "disclosedQuantity": 0,
            "price": float(target_price),
            "triggerPrice": float(round(target_price * 0.998, 2)),
            "price1": float(sl_price),
            "triggerPrice1": float(round(sl_price * 1.002, 2)),
            "quantity1": int(quantity)
        }
        try:
            r = requests.post(f"{self.BASE_URL}/forever/orders", json=payload, headers=self.headers, timeout=10)
            return r.json()
        except Exception as e:
            return {"error": str(e)}

@st.cache_data(ttl=3600)
def load_dhan_equity_master() -> pd.DataFrame:
    url = "https://images.dhan.co/api-data/api-scrip-master.csv"
    cols = ["SEM_EXM_EXCH_ID", "SEM_SEGMENT", "SEM_SMST_SECURITY_ID", "SEM_TRADING_SYMBOL", "SEM_INSTRUMENT_NAME"]
    try:
        df = pd.read_csv(url, usecols=cols, low_memory=False)
        df = df[(df["SEM_SEGMENT"] == "E") & (df["SEM_INSTRUMENT_NAME"] == "EQUITY")]
        df["CLEAN_SYM"] = df["SEM_TRADING_SYMBOL"].astype(str).str.upper().str.strip()
        return df
    except Exception:
        return pd.DataFrame()

def resolve_dhan_security_id(symbol: str, exchange: str = "NSE") -> str:
    master = load_dhan_equity_master()
    if master.empty:
        return "N/A"
    clean = symbol.replace(".NS", "").replace(".BO", "").upper().strip()
    sub = master[(master["SEM_EXM_EXCH_ID"] == exchange) & (master["CLEAN_SYM"] == clean)]
    if not sub.empty:
        return str(sub.iloc[0]["SEM_SMST_SECURITY_ID"])
    return "N/A"

# ==========================================
# 2. GLOBAL DHAN SIDEBAR CONNECTION
# ==========================================

st.sidebar.header("🟢 DhanHQ v2 Broker Link")
default_cid = st.secrets.get("DHAN_CLIENT_ID", "") if "DHAN_CLIENT_ID" in st.secrets else ""
default_tok = st.secrets.get("DHAN_ACCESS_TOKEN", "") if "DHAN_ACCESS_TOKEN" in st.secrets else ""

dhan_cid = st.sidebar.text_input("Dhan Client ID", value=default_cid, placeholder="e.g. 1100001234")
dhan_tok = st.sidebar.text_input("Dhan JWT Access Token", value=default_tok, type="password", placeholder="Paste 24h JWT Token")
global_mode = st.sidebar.radio("Execution Mode", ["Paper Swing Trade", "LIVE Dhan CNC Order"])

dhan_client = DhanHQClient(dhan_cid, dhan_tok) if (dhan_cid and dhan_tok) else None

if st.sidebar.button("🔌 Connect & Check Dhan Margin", use_container_width=True):
    if not dhan_client:
        st.sidebar.warning("Enter Client ID & JWT Access Token first.")
    else:
        funds = dhan_client.get_fund_limits()
        if "availabelBalance" in funds or "availableBalance" in funds or "dhanClientId" in funds:
            bal = funds.get("availabelBalance", funds.get("availableBalance", 0.0))
            st.session_state["dhan_connected"] = True
            st.session_state["dhan_balance"] = float(bal)
            st.sidebar.success(f"Connected! Margin: ₹{float(bal):,.2f}")
        else:
            st.session_state["dhan_connected"] = False
            st.sidebar.error(f"Token Error: {funds}")

if st.session_state.get("dhan_connected"):
    st.sidebar.metric("Available Dhan Margin (₹)", f"₹{st.session_state.get('dhan_balance', 0.0):,.2f}")

# ==========================================
# 3. NSE & BSE UNIVERSE & SMC ENGINE
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

def calculate_smc_features(df: pd.DataFrame, swing_len: int = 5) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA28"] = df["Close"].ewm(span=28, adjust=False).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = (100 - (100 / (1 + rs))).fillna(50)

    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean().bfill()

    df["Swing_High"] = df["High"].shift(1).rolling(swing_len * 2, min_periods=3).max()
    df["Swing_Low"] = df["Low"].shift(1).rolling(swing_len * 2, min_periods=3).min()

    df["Range_High"] = df["High"].rolling(40, min_periods=10).max()
    df["Range_Low"] = df["Low"].rolling(40, min_periods=10).min()
    df["Equilibrium_50"] = (df["Range_High"] + df["Range_Low"]) / 2.0
    df["In_Discount"] = df["Close"] <= df["Equilibrium_50"]

    df["Bullish_FVG"] = df["Low"] > df["High"].shift(2)
    df["FVG_Top"] = np.where(df["Bullish_FVG"], df["Low"], np.nan)
    df["FVG_Bottom"] = np.where(df["Bullish_FVG"], df["High"].shift(2), np.nan)
    df["Active_FVG_Top"] = pd.Series(df["FVG_Top"], index=df.index).ffill()
    df["Active_FVG_Bottom"] = pd.Series(df["FVG_Bottom"], index=df.index).ffill()

    prev_downtrend = df["EMA9"].shift(3) < df["EMA28"].shift(3)
    struct_break_up = (df["Close"] > df["Swing_High"]) & (df["Close"].shift(1) <= df["Swing_High"].shift(1))
    df["Bull_CHoCH"] = struct_break_up & prev_downtrend
    df["Bull_BOS"] = struct_break_up & (~prev_downtrend)

    df["Liq_Sweep_Bull"] = (df["Low"] < df["Swing_Low"]) & (df["Close"] > df["Swing_Low"])

    is_bearish = df["Close"] < df["Open"]
    impulsive_next = (df["Close"].shift(-1) - df["Open"].shift(-1)) > (1.1 * df["ATR"])
    df["Is_Bull_OB"] = is_bearish & impulsive_next
    df["OB_High"] = np.where(df["Is_Bull_OB"], df["High"], np.nan)
    df["OB_Low"] = np.where(df["Is_Bull_OB"], df["Low"], np.nan)
    df["Active_OB_High"] = pd.Series(df["OB_High"], index=df.index).ffill()
    df["Active_OB_Low"] = pd.Series(df["OB_Low"], index=df.index).ffill()

    df["OB_Mitigated"] = (df["Low"] <= df["Active_OB_High"] * 1.01) & (df["Close"] >= df["Active_OB_Low"])
    df["FVG_Mitigated"] = (df["Low"] <= df["Active_FVG_Top"] * 1.01) & (df["Close"] >= df["Active_FVG_Bottom"])
    return df

def score_stock_smc(symbol: str, df: pd.DataFrame) -> dict | None:
    if df is None or len(df) < 40:
        return None
    df = calculate_smc_features(df)
    last = df.iloc[-1]
    recent5 = df.iloc[-5:]

    close_p = float(last["Close"])
    atr = float(last["ATR"])
    rsi = float(last["RSI"])

    has_bos = bool(recent5["Bull_BOS"].any())
    has_choch = bool(recent5["Bull_CHoCH"].any())
    has_sweep = bool(recent5["Liq_Sweep_Bull"].any())
    has_fvg = bool(recent5["Bullish_FVG"].any() or last["FVG_Mitigated"])
    has_ob = bool(last["OB_Mitigated"])
    ema_bull = bool(last["EMA9"] > last["EMA28"])
    in_discount = bool(last["In_Discount"])

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

    ob_low = float(last["Active_OB_Low"]) if not np.isnan(last["Active_OB_Low"]) else (close_p - 1.5 * atr)
    struct_low = float(last["Swing_Low"]) if not np.isnan(last["Swing_Low"]) else (close_p - 1.5 * atr)
    sl_candidate = min(ob_low, struct_low) - (0.3 * atr)
    risk_per_share = float(np.clip(close_p - sl_candidate, close_p * 0.02, close_p * 0.065))
    sl_price = round(close_p - risk_per_share, 2)
    target_1 = round(close_p + (risk_per_share * 2.0), 2)
    target_2 = round(close_p + (risk_per_share * 3.2), 2)

    exch = "BSE" if symbol.endswith(".BO") else "NSE"
    clean_sym = symbol.replace(".NS", "").replace(".BO", "")
    sec_id = resolve_dhan_security_id(clean_sym, exch)

    return {
        "Symbol": clean_sym,
        "Exchange": exch,
        "Dhan Security ID": sec_id,
        "CMP (₹)": round(close_p, 2),
        "SMC Score": score,
        "SMC Setup": " + ".join(tags) if tags else "Neutral Structure",
        "Zone": "DISCOUNT" if in_discount else "PREMIUM",
        "Entry (₹)": round(close_p, 2),
        "Stop Loss (₹)": sl_price,
        "Target 1 (1:2) (₹)": target_1,
        "Target 2 (BSL) (₹)": target_2,
        "RSI (14)": round(rsi, 1),
        "Risk/Share (₹)": round(risk_per_share, 2)
    }

@st.cache_data(ttl=900)
def fetch_and_scan_universe(tickers: list[str]) -> pd.DataFrame:
    results = []
    data = yf.download(tickers, period="6mo", interval="1d", group_by="ticker", progress=False, threads=True)
    for t in tickers:
        try:
            sub_df = data[t].dropna(how="all") if len(tickers) > 1 else data.dropna(how="all")
            if not sub_df.empty and len(sub_df) >= 40:
                scored = score_stock_smc(t, sub_df)
                if scored:
                    results.append(scored)
        except Exception:
            continue
    out_df = pd.DataFrame(results)
    if not out_df.empty:
        out_df = out_df.sort_values(by=["SMC Score", "RSI (14)"], ascending=[False, False])
    return out_df

# ==========================================
# 4. SMC SWING BACKTESTER ENGINE
# ==========================================

def backtest_smc_swing(df: pd.DataFrame, capital: float = 100000.0, risk_pct: float = 1.5, rr_mult: float = 2.5) -> tuple[pd.DataFrame, dict]:
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
                setup_reason = "Bull CHoCH" if row["Bull_CHoCH"] else ("SSL Sweep" if row["Liq_Sweep_Bull"] else "BOS / OB Retest")
        else:
            exit_reason = None
            exit_p = close
            if low <= sl_p:
                exit_reason = "SMC Stop-Loss"
                exit_p = sl_p
            elif high >= tgt_p:
                exit_reason = f"Target Hit (1:{rr_mult})"
                exit_p = tgt_p
            elif close < float(row["Swing_Low"]) and (dt - pd.to_datetime(entry_date)).days >= 3:
                exit_reason = "Bearish CHoCH Exit"
                exit_p = close

            if exit_reason:
                pnl = round((exit_p - entry_p) * qty, 2)
                equity += pnl
                trades.append({
                    "Entry Date": entry_date, "Exit Date": dt.strftime("%Y-%m-%d"),
                    "SMC Trigger": setup_reason, "Qty": qty, "Entry (₹)": entry_p,
                    "Stop Loss (₹)": sl_p, "Exit (₹)": round(exit_p, 2),
                    "Exit Reason": exit_reason, "Net PnL (₹)": pnl, "Equity (₹)": round(equity, 2)
                })
                in_trade = False

    t_df = pd.DataFrame(trades)
    if t_df.empty:
        return t_df, {"Net Profit": 0.0, "Win Rate": 0.0, "Total Trades": 0, "Final Equity": capital}
    wins = t_df[t_df["Net PnL (₹)"] > 0]
    return t_df, {
        "Net Profit": round(equity - capital, 2),
        "Win Rate": round((len(wins) / len(t_df)) * 100.0, 1),
        "Total Trades": len(t_df),
        "Final Equity": round(equity, 2)
    }

# ==========================================
# 5. MAIN UI TABS
# ==========================================

st.title("🏦 SMC Institutional Swing Trader — Connected to DhanHQ v2")

if "paper_orders" not in st.session_state:
    st.session_state["paper_orders"] = []

tabs = st.tabs([
    "🔍 1. NSE/BSE SMC Scanner + 1-Click Dhan Order",
    "📐 2. SMC Structure & Order Block Chart",
    "📊 3. SMC Swing Backtester",
    "💼 4. Dhan Live Portfolio, Forever GTT & Orderbook"
])

# ------------------------------------------
# TAB 1: SCANNER + ONE-CLICK DHAN BRIDGE
# ------------------------------------------
with tabs[0]:
    c1, c2, c3, c4 = st.columns([1.8, 2.2, 1, 1])
    basket_choice = c1.selectbox("Select Market Universe", list(NSE_BSE_UNIVERSE.keys()))
    custom_tickers = c2.text_input("Custom NSE (.NS) or BSE (.BO) Symbols", value="ONGC.NS, SUZLON.NS, CUPID.NS, VPRPL.NS")
    min_score = c3.slider("Min SMC Score", 0, 80, 25, step=5)
    capital_size = c4.number_input("Swing Capital (₹)", min_value=10000, value=100000, step=10000)

    base_list = NSE_BSE_UNIVERSE[basket_choice]
    extra_list = [x.strip().upper() for x in custom_tickers.split(",") if x.strip()]
    full_tickers = list(dict.fromkeys(extra_list + base_list))

    if st.button("🚀 Scan NSE & BSE Market + Map Dhan Security IDs", type="primary", use_container_width=True):
        with st.spinner("Scanning market structure & resolving Dhan Security IDs..."):
            st.session_state["scan_df"] = fetch_and_scan_universe(full_tickers)

    if "scan_df" in st.session_state and not st.session_state["scan_df"].empty:
        df_f = st.session_state["scan_df"][st.session_state["scan_df"]["SMC Score"] >= min_score].copy()
        risk_rs = capital_size * 0.
