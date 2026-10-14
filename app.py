import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass, field
from typing import List, Optional, Literal
from datetime import time, datetime, timedelta

st.set_page_config(page_title="Personal AlgoTest Clone", layout="wide", page_icon="⚡")

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
    trail_sl_x: float = 0.0
    trail_sl_y: float = 0.0

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
    entry_price: Optional[float] = None
    entry_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_time: Optional[pd.Timestamp] = None
    exit_reason: Optional[str] = None
    sl_price: Optional[float] = None
    target_price: Optional[float] = None
    status: Literal["WAITING", "OPEN", "CLOSED"] = "WAITING"

# ==========================================
# 2. SYNTHETIC 1-MIN OPTIONS DATA GENERATOR
#    (Allows instant testing before plugging CSV/Parquet)
# ==========================================

@st.cache_data
def generate_sample_options_data(days: int = 30, underlying: str = "NIFTY") -> pd.DataFrame:
    np.random.seed(42)
    base_spot = 24500 if underlying == "NIFTY" else 52000
    step = 50 if underlying == "NIFTY" else 100
    records = []
    
    start_date = datetime(2026, 8, 1)
    trading_days = [start_date + timedelta(days=i) for i in range(days * 2) if (start_date + timedelta(days=i)).weekday() < 5][:days]

    for day in trading_days:
        spot = base_spot + np.random.randint(-3, 4) * step
        atm = int(round(spot / step) * step)
        strikes = [atm + i * step for i in range(-5, 6)]
        
        # 375 minutes from 09:15 to 15:29
        timestamps = [datetime.combine(day.date(), time(9, 15)) + timedelta(minutes=m) for m in range(375)]
        spot_moves = np.cumsum(np.random.normal(0, 4.5, len(timestamps)))
        
        for idx, ts in enumerate(timestamps):
            curr_spot = spot + spot_moves[idx]
            time_decay = (idx / 375.0) * 18.0  # Intraday theta decay simulation
            
            for k in strikes:
                for otype in ["CE", "PE"]:
                    intrinsic = max(0.0, curr_spot - k) if otype == "CE" else max(0.0, k - curr_spot)
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
# 3. INDIAN TAXES & BROKERAGE CALCULATOR
# ==========================================

def calculate_indian_charges(entry_price: float, exit_price: float, qty: int, action: str) -> float:
    buy_val = (entry_price if action == "BUY" else exit_price) * qty
    sell_val = (exit_price if action == "BUY" else entry_price) * qty
    turnover = buy_val + sell_val
    
    brokerage = 40.0  # ₹20 buy + ₹20 sell
    stt = sell_val * 0.001  # 0.1% on sell side of options
    txn_charges = turnover * 0.00035  # NSE Exchange Transaction Charges
    sebi_charges = turnover * 0.000001
    stamp_duty = buy_val * 0.00003
    gst = 0.18 * (brokerage + txn_charges + sebi_charges)
    
    return round(brokerage + stt + txn_charges + sebi_charges + stamp_duty + gst, 2)

# ==========================================
# 4. BACKTEST EXECUTION ENGINE
# ==========================================

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

            # Initialize at Entry Time
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
                        # Apply entry slippage
                        slip = 1 + (strategy.slippage_pct / 100.0) if leg.action == "BUY" else 1 - (strategy.slippage_pct / 100.0)
                        state.entry_price = round(init_p * slip, 2)
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

            # Evaluate Minute-by-Minute Candles
            mtm_pnl = 0.0
            for state in active_legs:
                c_rows = slice_df[(slice_df["strike"] == state.strike) & (slice_df["option_type"] == state.config.option_type)]
                if c_rows.empty:
                    continue
                candle = c_rows.iloc[0]
                high, low, close = float(candle["high"]), float(candle["low"]), float(candle["close"])

                # Wait & Trade Trigger
                if state.status == "WAITING":
                    trig = state.base_price * (1 + state.config.wait_and_trade_pct / 100.0)
                    if (state.config.wait_and_trade_pct > 0 and high >= trig) or (state.config.wait_and_trade_pct < 0 and low <= trig):
                        state.status = "OPEN"
                        state.entry_price = round(trig, 2)
                        state.entry_time = ts
                        dir_mult = 1 if state.config.action == "BUY" else -1
                        if state.config.sl_type == "PCT":
                            state.sl_price = round(state.entry_price * (1 - dir_mult * (state.config.sl_value / 100.0)), 2)
                        elif state.config.sl_type == "PTS":
                            state.sl_price = round(state.entry_price - dir_mult * state.config.sl_value, 2)

                # Active Leg Checks
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

            # Portfolio Level Exits
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
                        c_rows = slice_df[(slice_df["strike"] == state.strike) & (slice_df["option_type"] == state.config.option_type)]
                        close_p = float(c_rows.iloc[0]["close"]) if not c_rows.empty else state.entry_price
                        state.status = "CLOSED"
                        state.exit_price = close_p
                        state.exit_time = ts
                        state.exit_reason = force_reason
                break

        # EOD Accounting
        day_gross = 0.0
        day_charges = 0.0
        for s in active_legs:
            if s.entry_price and s.exit_price:
                qty = s.config.lots * strategy.lot_size
                mult = 1 if s.config.action == "BUY" else -1
                gross = (s.exit_price - s.entry_price) * mult * qty
                charges = calculate_indian_charges(s.entry_price, s.exit_price, qty, s.config.action) if strategy.include_charges else 0.0
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
# 5. STREAMLIT UI & VISUAL STRATEGY BUILDER
# ==========================================

st.title("⚡ Personal AlgoTest — Multi-Leg Options Backtester")

if "legs" not in st.session_state:
    st.session_state.legs = [
        {"action": "SELL", "option_type": "CE", "lots": 1, "strike_selection": "ATM_OFFSET", "strike_value": 0.0, "sl_type": "PCT", "sl_value": 25.0, "target_type": "NONE", "target_value": 0.0, "wait_pct": 0.0},
        {"action": "SELL", "option_type": "PE", "lots": 1, "strike_selection": "ATM_OFFSET", "strike_value": 0.0, "sl_type": "PCT", "sl_value": 25.0, "target_type": "NONE", "target_value": 0.0, "wait_pct": 0.0}
    ]

with st.container(border=True):
    st.subheader("1. Index & Timing Configuration")
    c1, c2, c3, c4, c5 = st.columns(5)
    underlying = c1.selectbox("Underlying Index", ["NIFTY", "BANKNIFTY"])
    lot_size = 75 if underlying == "NIFTY" else 30
    c1.caption(f"Lot Size: {lot_size}")
    entry_t = c2.time_input("Entry Time", value=time(9, 20))
    exit_t = c3.time_input("Exit Time", value=time(15, 15))
    sim_days = c4.slider("Backtest Horizon (Trading Days)", 10, 90, 30)
    data_source = c5.radio("Data Source", ["Built-in Simulator", "Upload CSV/Parquet"])

    uploaded_file = None
    if data_source == "Upload CSV/Parquet":
        uploaded_file = st.file_uploader("Upload 1-min Options Data (columns: date, timestamp, spot_close, strike, option_type, high, low, close)", type=["csv", "parquet"])

with st.container(border=True):
    st.subheader("2. Multi-Leg Builder")
    for idx, leg in enumerate(st.session_state.legs):
        cols = st.columns([1, 1, 1, 1.8, 1.2, 1.2, 1.2, 1.2, 1.2, 0.7])
        leg["action"] = cols[0].selectbox("B/S", ["SELL", "BUY"], index=0 if leg["action"] == "SELL" else 1, key=f"act_{idx}")
        leg["option_type"] = cols[1].selectbox("Type", ["CE", "PE"], index=0 if leg["option_type"] == "CE" else 1, key=f"opt_{idx}")
        leg["lots"] = cols[2].number_input("Lots", min_value=1, max_value=100, value=leg["lots"], key=f"lot_{idx}")
        leg["strike_selection"] = cols[3].selectbox("Strike Criterion", ["ATM_OFFSET", "CLOSEST_PREMIUM"], index=0 if leg["strike_selection"] == "ATM_OFFSET" else 1, key=f"sel_{idx}")
        leg["strike_value"] = cols[4].number_input("Offset / Prem", value=float(leg["strike_value"]), step=50.0, key=f"val_{idx}")
        leg["sl_type"] = cols[5].selectbox("SL Type", ["PCT", "PTS", "NONE"], index=["PCT", "PTS", "NONE"].index(leg["sl_type"]), key=f"slt_{idx}")
        leg["sl_value"] = cols[6].number_input("SL Val", value=float(leg["sl_value"]), step=5.0, key=f"slv_{idx}")
        leg["target_type"] = cols[7].selectbox("Tgt Type", ["NONE", "PCT", "PTS"], index=["NONE", "PCT", "PTS"].index(leg["target_type"]), key=f"tgtt_{idx}")
        leg["wait_pct"] = cols[8].number_input("W&T (%)", value=float(leg["wait_pct"]), step=5.0, key=f"wt_{idx}")
        if cols[9].button("🗑️", key=f"del_{idx}") and len(st.session_state.legs) > 1:
            st.session_state.legs.pop(idx)
            st.rerun()

    if st.button("➕ Add Option Leg"):
        st.session_state.legs.append({"action": "SELL", "option_type": "CE", "lots": 1, "strike_selection": "ATM_OFFSET", "strike_value": 0.0, "sl_type": "PCT", "sl_value": 25.0, "target_type": "NONE", "target_value": 0.0, "wait_pct": 0.0})
        st.rerun()

with st.container(border=True):
    st.subheader("3. Portfolio Risk & Execution Settings")
    r1, r2, r3, r4, r5 = st.columns(5)
    move_to_cost = r1.checkbox("Move SL to Cost on Leg SL", value=True)
    include_taxes = r2.checkbox("Deduct Brokerage & STT", value=True)
    overall_sl = r3.number_input("Overall MTM Stop Loss (₹)", min_value=0.0, value=4000.0, step=500.0)
    overall_tgt = r4.number_input("Overall MTM Target (₹)", min_value=0.0, value=0.0, step=500.0)
    slippage = r5.number_input("Slippage (%)", min_value=0.0, max_value=5.0, value=0.5, step=0.1)

# ==========================================
# 6. RUN BACKTEST & RENDER ALGOTEST REPORT
# ==========================================

if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
    if data_source == "Upload CSV/Parquet" and uploaded_file is not None:
        df_data = pd.read_parquet(uploaded_file) if uploaded_file.name.endswith(".parquet") else pd.read_csv(uploaded_file, parse_dates=["timestamp", "date"])
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

    # AlgoTest KPI Metrics
    total_net = daily_df["Net PnL"].sum()
    max_dd = daily_df["Drawdown"].min()
    wins = daily_df[daily_df["Net PnL"] > 0]
    losses = daily_df[daily_df["Net PnL"] <= 0]
    win_rate = (len(wins) / len(daily_df)) * 100 if len(daily_df) > 0 else 0
    avg_win = wins["Net PnL"].mean() if not wins.empty else 0
    avg_loss = abs(losses["Net PnL"].mean()) if not losses.empty else 0
    expectancy = ((win_rate / 100.0 * avg_win) - ((1 - win_rate / 100.0) * avg_loss)) / avg_loss if avg_loss > 0 else 0
    ret_mdd = abs(total_net / max_dd) if max_dd != 0 else 0

    st.subheader("📊 Strategy Report Card")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Net Profit (₹)", f"₹{total_net:,.2f}", delta=f"Gross: ₹{daily_df['Gross PnL'].sum():,.0f}")
    m2.metric("Win Rate",