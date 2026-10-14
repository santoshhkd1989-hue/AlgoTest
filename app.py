import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from dataclasses import dataclass, field
from typing import List, Optional, Literal
from datetime import time, datetime, timedelta

st.set_page_config(page_title="Personal AlgoTest — Dhan Edition", layout="wide", page_icon="⚡")

# ==========================================
# 1. CORE ENGINE & DATA STRUCTURES
# ==========================================

@dataclass
class OptionLeg:
    leg_id: int
    action: Literal["BUY", "SELL"]
    option_type: Literal["CE", "PE"]
    lots: int = 1
    strike_selection: Literal["ATM_OFFSET", "CLOSEST_PREMIUM"] = "ATM_OFFSET"
    strike_value: float = 0.0
    wait_and_trade_pct: float = 0.0
    sl_type: Literal["PCT", "PTS", "NONE"] = "PCT"
    sl_value: float = 25.0
    target_type: Literal["PCT", "PTS", "NONE"] = "NONE"
    target_value: float = 0.0

@dataclass
class StrategyConfig:
    name: str
    underlying: str = "NIFTY"
    lot_size: int = 75
    entry_time: time = time(9, 20)
    exit_time: time = time(15, 15)
    legs: List[OptionLeg] = field(default_factory=list)
    move_sl_to_cost: bool = False
    overall_sl_mtm: float = 0.0
    overall_target_mtm: float = 0.0
    slippage_pct: float = 0.5
    include_charges: bool = True

@dataclass
class ActiveLegState:
    config: OptionLeg
    strike: int
    base_price: float
    security_id: str = ""
    entry_price: Optional[float] = None
    entry_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_time: Optional[pd.Timestamp] = None
    exit_reason: Optional[str] = None
    sl_price: Optional[float] = None
    target_price: Optional[float] = None
    status: Literal["WAITING", "OPEN", "CLOSED"] = "WAITING"

# ==========================================
# 2. DHANHQ V2 API CLIENT (BUILT-IN)
# ==========================================

class DhanBrokerClient:
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

    def get_fund_limits(self) -> dict:
        resp = requests.get(f"{self.BASE_URL}/fundlimit", headers=self.headers, timeout=10)
        return resp.json() if resp.status_code == 200 else {"error": resp.text}

    def get_positions(self) -> dict:
        resp = requests.get(f"{self.BASE_URL}/positions", headers=self.headers, timeout=10)
        return resp.json() if resp.status_code == 200 else {"error": resp.text}

    def get_orders(self) -> dict:
        resp = requests.get(f"{self.BASE_URL}/orders", headers=self.headers, timeout=10)
        return resp.json() if resp.status_code == 200 else {"error": resp.text}

    def place_order(
        self,
        security_id: str,
        transaction_type: str,
        quantity: int,
        order_type: str = "MARKET",
        price: float = 0.0,
        trigger_price: float = 0.0
    ) -> dict:
        payload = {
            "dhanClientId": self.client_id,
            "transactionType": transaction_type,
            "exchangeSegment": "NSE_FNO",
            "productType": "INTRADAY",
            "orderType": order_type,
            "validity": "DAY",
            "securityId": str(security_id),
            "quantity": int(quantity),
            "disclosedQuantity": 0,
            "price": float(price),
            "triggerPrice": float(trigger_price),
            "afterMarketOrder": False
        }
        resp = requests.post(f"{self.BASE_URL}/orders", json=payload, headers=self.headers, timeout=10)
        return resp.json()

    def get_option_chain(self, underlying_scrip: int, expiry_date: str) -> dict:
        # Nifty 50 Index Security ID = 13, BankNifty = 25 in IDX_I
        payload = {
            "UnderlyingScrip": underlying_scrip,
            "UnderlyingSeg": "IDX_I",
            "Expiry": expiry_date
        }
        resp = requests.post(f"{self.BASE_URL}/optionchain", json=payload, headers=self.headers, timeout=10)
        return resp.json()

@st.cache_data(ttl=3600)
def load_dhan_fno_master() -> pd.DataFrame:
    """Downloads Dhan's official compact instrument master and filters NSE F&O options."""
    url = "https://images.dhan.co/api-data/api-scrip-master.csv"
    cols = [
        "SEM_EXM_EXCH_ID", "SEM_SEGMENT", "SEM_SMST_SECURITY_ID",
        "SEM_INSTRUMENT_NAME", "SEM_TRADING_SYMBOL", "SEM_LOT_UNITS",
        "SEM_EXPIRY_DATE", "SEM_STRIKE_PRICE", "SEM_OPTION_TYPE"
    ]
    try:
        df = pd.read_csv(url, usecols=cols, low_memory=False)
        df = df[(df["SEM_EXM_EXCH_ID"] == "NSE") & (df["SEM_SEGMENT"] == "D") & (df["SEM_INSTRUMENT_NAME"] == "OPTIDX")]
        df["SEM_STRIKE_PRICE"] = pd.to_numeric(df["SEM_STRIKE_PRICE"], errors="coerce").fillna(0).astype(int)
        return df
    except Exception:
        return pd.DataFrame()

# ==========================================
# 3. SYNTHETIC 1-MIN OPTIONS DATA GENERATOR
# ==========================================

@st.cache_data
def generate_sample_options_data(days: int = 30, underlying: str = "NIFTY") -> pd.DataFrame:
    np.random.seed(42)
    base_spot = 24500 if underlying == "NIFTY" else 52000
    step = 50 if underlying == "NIFTY" else 100
    records = []
    
    start_date = datetime(2026, 8, 1)
    trading_days = [
        start_date + timedelta(days=i)
        for i in range(days * 2)
        if (start_date + timedelta(days=i)).weekday() < 5
    ][:days]

    for day in trading_days:
        spot = base_spot + np.random.randint(-3, 4) * step
        atm = int(round(spot / step) * step)
        strikes = [atm + i * step for i in range(-5, 6)]
        
        timestamps = [
            datetime.combine(day.date(), time(9, 15)) + timedelta(minutes=m)
            for m in range(375)
        ]
        spot_moves = np.cumsum(np.random.normal(0, 4.5, len(timestamps)))
        
        for idx, ts in enumerate(timestamps):
            curr_spot = spot + spot_moves[idx]
            time_decay = (idx / 375.0) * 18.0
            
            for k in strikes:
                for otype in ["CE", "PE"]:
                    if otype == "CE":
                        intrinsic = max(0.0, curr_spot - k)
                    else:
                        intrinsic = max(0.0, k - curr_spot)
                    dist = abs(curr_spot - k)
                    extrinsic = max(5.0, 120.0 * np.exp(-dist / 350.0) - time_decay)
                    close_p = max(1.0, round(intrinsic + extrinsic + np.random.normal(0, 1.2), 2))
                    high_p = round(close_p + abs(np.random.normal(0, 2.5)), 2)
                    low_p = max(0.5, round(close_p - abs(np.random.normal(0, 2.5)), 2))
                    
                    records.append({
                        "date": day.date(),
                        "timestamp": ts,
                        "spot_close": round(curr_spot, 2),
                        "strike": k,
                        "option_type": otype,
                        "open": close_p,
                        "high": high_p,
                        "low": low_p,
                        "close": close_p
                    })
    return pd.DataFrame(records)

# ==========================================
# 4. INDIAN TAXES & BACKTEST ENGINE
# ==========================================

def calculate_indian_charges(entry_price: float, exit_price: float, qty: int, action: str) -> float:
    buy_val = (entry_price if action == "BUY" else exit_price) * qty
    sell_val = (exit_price if action == "BUY" else entry_price) * qty
    turnover = buy_val + sell_val
    
    brokerage = 40.0
    stt = sell_val * 0.001
    txn_charges = turnover * 0.00035
    sebi_charges = turnover * 0.000001
    stamp_duty = buy_val * 0.00003
    gst = 0.18 * (brokerage + txn_charges + sebi_charges)
    
    return round(brokerage + stt + txn_charges + sebi_charges + stamp_duty + gst, 2)

def run_backtest(df: pd.DataFrame, strategy: StrategyConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    step = 50 if strategy.underlying == "NIFTY" else 100
    daily_results = []
    trade_logs = []

    for day_date, day_df in df.groupby("date"):
        timestamps = sorted(day_df["timestamp"].unique())
        active_legs: List[ActiveLegState] = []
        initialized = False

        for ts in timestamps:
            curr_time = ts.time()
            if curr_time < strategy.entry_time:
                continue

            slice_df = day_df[day_df["timestamp"] == ts]
            spot_price = float(slice_df["spot_close"].iloc[0])

            if not initialized:
                for leg in strategy.legs:
                    opt_df = slice_df[slice_df["option_type"] == leg.option_type]
                    if leg.strike_selection == "ATM_OFFSET":
                        atm = int(round(spot_price / step) * step)
                        target_k = int(atm + leg.strike_value)
                        row = opt_df.iloc[(opt_df["strike"] - target_k).abs().argsort()[:1]].iloc[0]
                    else:
                        row = opt_df.iloc[(opt_df["close"] - leg.strike_value).abs().argsort()[:1]].iloc[0]

                    strike = int(row["strike"])
                    init_p = float(row["close"])
                    
                    state = ActiveLegState(config=leg, strike=strike, base_price=init_p)
                    if leg.wait_and_trade_pct == 0.0:
                        state.status = "OPEN"
                        slip_mult = 1 + (strategy.slippage_pct / 100.0) if leg.action == "BUY" else 1 - (strategy.slippage_pct / 100.0)
                        state.entry_price = round(init_p * slip_mult, 2)
                        state.entry_time = ts
                        dir_mult = 1 if leg.action == "BUY" else -1
                        if leg.sl_type == "PCT":
                            state.sl_price = round(state.entry_price * (1 - dir_mult * (leg.sl_value / 100.0)), 2)
                        elif leg.sl_type == "PTS":
                            state.sl_price = round(state.entry_price - dir_mult * leg.sl_value, 2)
                        if leg.target_type == "PCT":
                            state.target_price = round(state.entry_price * (1 + dir_mult * (leg.target_value / 100.0)), 2)
                        elif leg.target_type == "PTS":
                            state.target_price = round(state.entry_price + dir_mult * leg.target_value, 2)
                    active_legs.append(state)
                initialized = True
                continue

            mtm_pnl = 0.0
            for state in active_legs:
                c_rows = slice_df[
                    (slice_df["strike"] == state.strike) &
                    (slice_df["option_type"] == state.config.option_type)
                ]
                if c_rows.empty:
                    continue
                candle = c_rows.iloc[0]
                high, low, close = float(candle["high"]), float(candle["low"]), float(candle["close"])

                if state.status == "WAITING":
                    trig = state.base_price * (1 + state.config.wait_and_trade_pct / 100.0)
                    cond_up = state.config.wait_and_trade_pct > 0 and high >= trig
                    cond_dn = state.config.wait_and_trade_pct < 0 and low <= trig
                    if cond_up or cond_dn:
                        state.status = "OPEN"
                        state.entry_price = round(trig, 2)
                        state.entry_time = ts
                        dir_mult = 1 if state.config.action == "BUY" else -1
                        if state.config.sl_type == "PCT":
                            state.sl_price = round(state.entry_price * (1 - dir_mult * (state.config.sl_value / 100.0)), 2)
                        elif state.config.sl_type == "PTS":
                            state.sl_price = round(state.entry_price - dir_mult * state.config.sl_value, 2)

                if state.status == "OPEN":
                    is_buy = state.config.action == "BUY"
                    if state.sl_price is not None:
                        if (is_buy and low <= state.sl_price) or (not is_buy and high >= state.sl_price):
                            state.status = "CLOSED"
                            state.exit_price = state.sl_price
                            state.exit_time = ts
                            state.exit_reason = "SL_HIT"
                            if strategy.move_sl_to_cost:
                                for other in active_legs:
                                    if other.status == "OPEN" and other.config.leg_id != state.config.leg_id:
                                        other.sl_price = other.entry_price

                    if state.status == "OPEN" and state.target_price is not None:
                        if (is_buy and high >= state.target_price) or (not is_buy and low <= state.target_price):
                            state.status = "CLOSED"
                            state.exit_price = state.target_price
                            state.exit_time = ts
                            state.exit_reason = "TARGET_HIT"

                if state.entry_price is not None:
                    curr_exit = state.exit_price if state.status == "CLOSED" else close
                    mult = 1 if state.config.action == "BUY" else -1
                    mtm_pnl += (curr_exit - state.entry_price) * mult * state.config.lots * strategy.lot_size

            force_reason = None
            if strategy.overall_sl_mtm > 0 and mtm_pnl <= -abs(strategy.overall_sl_mtm):
                force_reason = "OVERALL_SL"
            elif strategy.overall_target_mtm > 0 and mtm_pnl >= abs(strategy.overall_target_mtm):
                force_reason = "OVERALL_TARGET"
            elif curr_time >= strategy.exit_time:
                force_reason = "TIME_EXIT"

            if force_reason:
                for state in active_legs:
                    if state.status == "OPEN":
                        c_rows = slice_df[
                            (slice_df["strike"] == state.strike) &
                            (slice_df["option_type"] == state.config.option_type)
                        ]
                        close_p = float(c_rows.iloc[0]["close"]) if not c_rows.empty else state.entry_price
                        state.status = "CLOSED"
                        state.exit_price = close_p
                        state.exit_time = ts
                        state.exit_reason = force_reason
                break

        day_gross = 0.0
        day_charges = 0.0
        for s in active_legs:
            if s.entry_price and s.exit_price:
                qty = s.config.lots * strategy.lot_size
                mult = 1 if s.config.action == "BUY" else -1
                gross = (s.exit_price - s.entry_price) * mult * qty
                if strategy.include_charges:
                    charges = calculate_indian_charges(s.entry_price, s.exit_price, qty, s.config.action)
                else:
                    charges = 0.0
                net = gross - charges
                day_gross += gross
                day_charges += charges
                trade_logs.append({
                    "Date": day_date,
                    "Leg": f"Leg {s.config.leg_id} ({s.config.action} {s.strike} {s.config.option_type})",
                    "Entry Time": s.entry_time.strftime("%H:%M") if s.entry_time else "-",
                    "Entry (₹)": s.entry_price,
                    "Exit Time": s.exit_time.strftime("%H:%M") if s.exit_time else "-",
                    "Exit (₹)": s.exit_price,
                    "Reason": s.exit_reason,
                    "Gross PnL (₹)": round(gross, 2),
                    "Charges (₹)": round(charges, 2),
                    "Net PnL (₹)": round(net, 2)
                })

        daily_results.append({
            "Date": pd.to_datetime(day_date),
            "Gross PnL": round(day_gross, 2),
            "Charges": round(day_charges, 2),
            "Net PnL": round(day_gross - day_charges, 2)
        })

    return pd.DataFrame(daily_results), pd.DataFrame(trade_logs)

# ==========================================
# 5. SHARED STRATEGY BUILDER UI
# ==========================================

st.title("⚡ Personal AlgoTest — Strategy Builder & Dhan Execution")

if "legs" not in st.session_state:
    st.session_state.legs = [
        {
            "action": "SELL", "option_type": "CE", "lots": 1,
            "strike_selection": "ATM_OFFSET", "strike_value": 0.0,
            "sl_type": "PCT", "sl_value": 25.0,
            "target_type": "NONE", "target_value": 0.0, "wait_pct": 0.0
        },
        {
            "action": "SELL", "option_type": "PE", "lots": 1,
            "strike_selection": "ATM_OFFSET", "strike_value": 0.0,
            "sl_type": "PCT", "sl_value": 25.0,
            "target_type": "NONE", "target_value": 0.0, "wait_pct": 0.0
        }
    ]

with st.container(border=True):
    st.subheader("1. Strategy & Multi-Leg Setup")
    c1, c2, c3, c4 = st.columns(4)
    underlying = c1.selectbox("Underlying Index", ["NIFTY", "BANKNIFTY"])
    lot_size = 75 if underlying == "NIFTY" else 30
    c1.caption(f"Standard Lot Size: {lot_size}")
    entry_t = c2.time_input("Entry Time", value=time(9, 20))
    exit_t = c3.time_input("Exit Time", value=time(15, 15))
    move_to_cost = c4.checkbox("Move SL to Cost on Leg SL", value=True)

    st.divider()
    for idx, leg in enumerate(st.session_state.legs):
        cols = st.columns([1, 1, 1, 1.8, 1.2, 1.2, 1.2, 1.2, 1.2, 0.7])
        act_idx = 0 if leg["action"] == "SELL" else 1
        opt_idx = 0 if leg["option_type"] == "CE" else 1
        sel_idx = 0 if leg["strike_selection"] == "ATM_OFFSET" else 1
        slt_idx = ["PCT", "PTS", "NONE"].index(leg["sl_type"])
        tgt_idx = ["NONE", "PCT", "PTS"].index(leg["target_type"])

        leg["action"] = cols[0].selectbox("B/S", ["SELL", "BUY"], index=act_idx, key=f"act_{idx}")
        leg["option_type"] = cols[1].selectbox("Type", ["CE", "PE"], index=opt_idx, key=f"opt_{idx}")
        leg["lots"] = cols[2].number_input("Lots", 1, 100, value=leg["lots"], key=f"lot_{idx}")
        leg["strike_selection"] = cols[3].selectbox("Criterion", ["ATM_OFFSET", "CLOSEST_PREMIUM"], index=sel_idx, key=f"sel_{idx}")
        leg["strike_value"] = cols[4].number_input("Offset/Prem", value=float(leg["strike_value"]), step=50.0, key=f"val_{idx}")
        leg["sl_type"] = cols[5].selectbox("SL Type", ["PCT", "PTS", "NONE"], index=slt_idx, key=f"slt_{idx}")
        leg["sl_value"] = cols[6].number_input("SL Val", value=float(leg["sl_value"]), step=5.0, key=f"slv_{idx}")
        leg["target_type"] = cols[7].selectbox("Tgt Type", ["NONE", "PCT", "PTS"], index=tgt_idx, key=f"tgtt_{idx}")
        leg["wait_pct"] = cols[8].number_input("W&T (%)", value=float(leg["wait_pct"]), step=5.0, key=f"wt_{idx}")
        if cols[9].button("🗑️", key=f"del_{idx}") and len(st.session_state.legs) > 1:
            st.session_state.legs.pop(idx)
            st.rerun()

    if st.button("➕ Add Option Leg"):
        st.session_state.legs.append({
            "action": "SELL", "option_type": "CE", "lots": 1,
            "strike_selection": "ATM_OFFSET", "strike_value": 0.0,
            "sl_type": "PCT", "sl_value": 25.0,
            "target_type": "NONE", "target_value": 0.0, "wait_pct": 0.0
        })
        st.rerun()

    r1, r2, r3, r4 = st.columns(4)
    overall_sl = r1.number_input("Overall MTM Stop Loss (₹)", min_value=0.0, value=4000.0, step=500.0)
    overall_tgt = r2.number_input("Overall MTM Target (₹)", min_value=0.0, value=0.0, step=500.0)
    slippage = r3.number_input("Slippage (%)", min_value=0.0, max_value=5.0, value=0.5, step=0.1)
    include_taxes = r4.checkbox("Deduct Brokerage & STT", value=True)

# ==========================================
# 6. TABS: BACKTESTER VS. DHAN LIVE EXECUTION
# ==========================================

tab_bt, tab_dhan = st.tabs(["📊 Historical Backtester", "🟢 Dhan Broker Live & Paper Trading"])

with tab_bt:
    b1, b2 = st.columns([1, 2])
    sim_days = b1.slider("Backtest Horizon (Trading Days)", 10, 90, 30)
    data_source = b2.radio("Historical Data Source", ["Built-in Simulator", "Upload CSV/Parquet"], horizontal=True)
    uploaded_file = None
    if data_source == "Upload CSV/Parquet":
        uploaded_file = st.file_uploader("Upload 1-min Options Data", type=["csv", "parquet"])

    if st.button("🚀 Run Strategy Backtest", type="primary", use_container_width=True):
        if data_source == "Upload CSV/Parquet" and uploaded_file is not None:
            if uploaded_file.name.endswith(".parquet"):
                df_data = pd.read_parquet(uploaded_file)
            else:
                df_data = pd.read_csv(uploaded_file, parse_dates=["timestamp", "date"])
        else:
            df_data = generate_sample_options_data(days=sim_days, underlying=underlying)

        leg_objs = [
            OptionLeg(
                leg_id=i + 1, action=l["action"], option_type=l["option_type"], lots=l["lots"],
                strike_selection=l["strike_selection"], strike_value=l["strike_value"],
                wait_and_trade_pct=l["wait_pct"], sl_type=l["sl_type"], sl_value=l["sl_value"],
                target_type=l["target_type"], target_value=l["target_value"]
            ) for i, l in enumerate(st.session_state.legs)
        ]

        cfg = StrategyConfig(
            name="Custom Strategy", underlying=underlying, lot_size=lot_size,
            entry_time=entry_t, exit_time=exit_t, legs=leg_objs,
            move_sl_to_cost=move_to_cost, overall_sl_mtm=overall_sl,
            overall_target_mtm=overall_tgt, slippage_pct=slippage, include_charges=include_taxes
        )

        daily_df, trades_df = run_backtest(df_data, cfg)
        daily_df["Cumulative Net PnL"] = daily_df["Net PnL"].cumsum()
        daily_df["Peak"] = daily_df["Cumulative Net PnL"].cummax()
        daily_df["Drawdown"] = daily_df["Cumulative Net PnL"] - daily_df["Peak"]

        total_net = daily_df["Net PnL"].sum()
        gross_sum = daily_df["Gross PnL"].sum()
        charges_sum = daily_df["Charges"].sum()
        max_dd = daily_df["Drawdown"].min()
        wins = daily_df[daily_df["Net PnL"] > 0]
        losses = daily_df[daily_df["Net PnL"] <= 0]
        win_rate = (len(wins) / len(daily_df)) * 100 if len(daily_df) > 0 else 0.0
        avg_win = wins["Net PnL"].mean() if not wins.empty else 0.0
        avg_loss = abs(losses["Net PnL"].mean()) if not losses.empty else 0.0
        expectancy = ((win_rate / 100.0 * avg_win) - ((1 - win_rate / 100.0) * avg_loss)) / avg_loss if avg_loss > 0 else 0.0
        ret_mdd = abs(total_net / max_dd) if max_dd != 0 else 0.0

        st.subheader("📊 Strategy Report Card")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Net Profit (₹)", f"₹{total_net:,.2f}", delta=f"Gross: ₹{gross_sum:,.0f}")
        m2.metric("Win Rate", f"{win_rate:.1f}%", delta=f"{len(wins)}W / {len(losses)}L")
        m3.metric("Max Drawdown", f"₹{max_dd:,.2f}")
        m4.metric("Return / MDD", f"{ret_mdd:.2f}x")
        m5.metric("Expectancy", f"{expectancy:.2f}")
        m6.metric("Total Charges", f"₹{charges_sum:,.2f}")

        c_left, c_right = st.columns(2)
        with c_left:
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                x=daily_df["Date"], y=daily_df["Cumulative Net PnL"],
                mode="lines+markers", name="Net Equity", fill="tozeroy",
                line=dict(color="#10b981", width=2)
            ))
            fig_eq.update_layout(title="Cumulative Net PnL Curve (₹)", height=360)
            st.plotly_chart(fig_eq, use_container_width=True)

        with c_right:
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=daily_df["Date"], y=daily_df["Drawdown"],
                mode="lines", name="Drawdown", fill="tozeroy",
                line=dict(color="#ef4444", width=2)
            ))
            fig_dd.update_layout(title="Underwater Drawdown Chart (₹)", height=360)
            st.plotly_chart(fig_dd, use_container_width=True)

        st.subheader("📜 Detailed Leg-Wise Trade Log")
        st.dataframe(trades_df, use_container_width=True, hide_index=True)

with tab_dhan:
    st.subheader("🟢 DhanHQ v2 Live Execution & Forward Testing")
    st.caption("Get your Client ID & 24-hour Access Token from web.dhan.co -> My Profile -> Access DhanHQ APIs")

    d1, d2, d3 = st.columns([1.5, 2.5, 1.5])
    dhan_client_id = d1.text_input("Dhan Client ID", placeholder="e.g. 1100001234")
    dhan_token = d2.text_input("Dhan JWT Access Token", type="password", placeholder="Paste your Dhan v2 Access Token")
    exec_mode = d3.radio("Execution Mode", ["Paper Trade (Forward Test)", "LIVE Real Money Orders"])

    client = DhanBrokerClient(dhan_client_id, dhan_token) if dhan_client_id and dhan_token else None

    col_btn1, col_btn2, col_btn3 = st.columns(3)
    if col_btn1.button("🔌 Verify Dhan Connection & Margin", use_container_width=True):
        if not client:
            st.warning("Please enter both your Dhan Client ID and Access Token.")
        else:
            funds = client.get_fund_limits()
            if "availabelBalance" in funds or "dhanClientId" in funds:
                avail = funds.get("availabelBalance", funds.get("availableBalance", 0))
                st.success(f"Connected to Dhan Account {dhan_client_id}! Available Margin: ₹{avail:,.2f}")
                st.json(funds)
            else:
                st.error(f"Connection Failed: {funds}")

    if col_btn2.button("⚡ Deploy Strategy Legs Now", type="primary", use_container_width=True):
        st.info(f"Deploying {len(st.session_state.legs)} legs in **{exec_mode}** mode...")
        master_df = load_dhan_fno_master()
        step = 50 if underlying == "NIFTY" else 100
        ref_spot = 24500 if underlying == "NIFTY" else 52000
        atm_strike = int(round(ref_spot / step) * step)

        order_results = []
        for i, leg in enumerate(st.session_state.legs):
            target_strike = int(atm_strike + leg["strike_value"]) if leg["strike_selection"] == "ATM_OFFSET" else atm_strike
            qty = int(leg["lots"] * lot_size)
            sec_id = "N/A"
            symbol_name = f"{underlying} {target_strike} {leg['option_type']}"

            if not master_df.empty:
                sub = master_df[
                    (master_df["SEM_TRADING_SYMBOL"].str.startswith(underlying)) &
                    (master_df["SEM_STRIKE_PRICE"] == target_strike) &
                    (master_df["SEM_OPTION_TYPE"] == leg["option_type"])
                ].sort_values("SEM_EXPIRY_DATE")
                if not sub.empty:
                    sec_id = str(sub.iloc[0]["SEM_SMST_SECURITY_ID"])
                    symbol_name = str(sub.iloc[0]["SEM_TRADING_SYMBOL"])

            if exec_mode == "Paper Trade (Forward Test)":
                order_results.append({
                    "Leg": i + 1,
                    "Mode": "PAPER",
                    "Symbol": symbol_name,
                    "Dhan Security ID": sec_id,
                    "Action": leg["action"],
                    "Qty": qty,
                    "Status": "SIMULATED_FILLED"
                })
            else:
                if not client:
                    st.error("Enter Dhan credentials before firing LIVE orders.")
                    break
                resp = client.place_order(
                    security_id=sec_id,
                    transaction_type=leg["action"],
                    quantity=qty,
                    order_type="MARKET"
                )
                order_results.append({
                    "Leg": i + 1,
                    "Mode": "LIVE",
                    "Symbol": symbol_name,
                    "Dhan Security ID": sec_id,
                    "Action": leg["action"],
                    "Qty": qty,
                    "API Response": str(resp)
                })

        if order_results:
            st.dataframe(pd.DataFrame(order_results), use_container_width=True, hide_index=True)

    if col_btn3.button("🚨 Fetch Live Positions / Orderbook", use_container_width=True):
        if not client:
            st.warning("Enter your Dhan credentials first.")
        else:
            pos = client.get_positions()
            ords = client.get_orders()
            st.write("**Open Dhan Positions:**", pos)
            st.write("**Today's Dhan Orders:**", ords)
