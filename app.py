import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

# ============================================================================
# SAFE PLOTLY IMPORT (CRITICAL FIX)
# ============================================================================
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError as e:
    st.error("❌ Plotly module not found. Please ensure plotly is in requirements.txt")
    st.error(f"Error: {str(e)}")
    PLOTLY_AVAILABLE = False
    st.stop()

try:
    from smartapi import SmartConnect
except ImportError:
    st.error("❌ SmartAPI not found. Add 'smartapi-python==1.3.7' to requirements.txt")
    st.stop()

try:
    from ta.momentum import RSIIndicator
except ImportError:
    st.error("❌ TA library not found. Add 'ta==0.11.0' to requirements.txt")
    st.stop()

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="AstroQuant Pro V10.0 - SmartAPI Edition",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# SBC ENGINE (NAKSHATRA + TITHI)
# ============================================================================

NAKSHATRAS = [
    "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigasira", "Ardra",
    "Punarvasu", "Pushya", "Aslesha", "Magha", "Purva Phalguni",
    "Uttara Phalguni", "Hasta", "Chitra", "Svati", "Visakha",
    "Anuradha", "Jyestha", "Mula", "Purva Ashadha", "Uttara Ashadha",
    "Sravana", "Dhanistha", "Shatabhisha", "Purva Bhadrapada",
    "Uttara Bhadrapada", "Revati"
]

NAKSHATRA_SENTIMENT = {
    "Pushya": "Reversal",
    "Ashwini": "Bullish", "Rohini": "Bullish", "Hasta": "Bullish",
    "Chitra": "Bullish", "Anuradha": "Bullish", "Sravana": "Bullish", "Revati": "Bullish",
    "Bharani": "Bearish", "Krittika": "Bearish", "Ardra": "Bearish",
    "Aslesha": "Bearish", "Jyestha": "Bearish", "Mula": "Bearish", "Shatabhisha": "Bearish"
}

def get_market_nakshatra(price: float) -> str:
    if price <= 0:
        return "Ashwini"
    idx = int(price / 100) % 27
    return NAKSHATRAS[idx]

def get_tithi_info():
    day_of_month = datetime.now().day
    tithi = ((day_of_month - 1) % 30) + 1
    tithi_mapping = {
        "Nanda": [1, 6, 11, 16, 21, 26],
        "Bhadra": [2, 7, 12, 17, 22, 27],
        "Jaya": [3, 8, 13, 18, 23, 28],
        "Rikta": [4, 9, 14, 19, 24, 29],
        "Poorna": [5, 10, 15, 20, 25, 30],
    }
    for name, days in tithi_mapping.items():
        if tithi in days:
            return name, tithi
    return "Unknown", tithi

def compute_sbc_score(sbc_context):
    score = 0
    reasons = []

    for label, nak in [("Moon", sbc_context.get("moon_nakshatra")),
                       ("Index", sbc_context.get("index_nakshatra"))]:
        if not nak:
            continue
        sentiment = NAKSHATRA_SENTIMENT.get(nak)
        if sentiment == "Bullish":
            score += 10
            reasons.append(f"✓ {label} in bullish nakshatra {nak} (+10)")
        elif sentiment == "Bearish":
            score -= 10
            reasons.append(f"✗ {label} in bearish nakshatra {nak} (−10)")
        elif sentiment == "Reversal":
            score += 15
            reasons.append(f"★ {label} in Pushya (major reversal zone) (+15)")

    tithi_name = sbc_context.get("tithi_name")
    if tithi_name == "Nanda":
        score += 10
        reasons.append("✓ Nanda tithi (prosperity/growth) (+10)")
    elif tithi_name == "Jaya":
        score += 5
        reasons.append("~ Jaya tithi (victory/trending) (+5)")
    elif tithi_name == "Rikta":
        score -= 20
        reasons.append("✗ Rikta tithi (empty/volatile – avoid aggression) (−20)")
    elif tithi_name == "Poorna":
        score += 5
        reasons.append("~ Poorna tithi (completion/strong moves) (+5)")

    return score, reasons

# ============================================================================
# OPTION CHAIN ENGINE
# ============================================================================

class OptionChainAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.spot_price = None
        self.max_pain = None
        self.pcr = None

    def calculate_max_pain(self):
        try:
            strikes = sorted(self.df["Strike Price"].unique())
            min_pain = float("inf")
            max_pain_strike = strikes[len(strikes) // 2]
            for strike in strikes:
                call_pain = 0
                put_pain = 0
                for _, row in self.df.iterrows():
                    s = row["Strike Price"]
                    if s < strike:
                        call_pain += row["Call OI"] * (strike - s)
                    if s > strike:
                        put_pain += row["Put OI"] * (s - strike)
                total_pain = call_pain + put_pain
                if total_pain < min_pain:
                    min_pain = total_pain
                    max_pain_strike = strike
            self.max_pain = max_pain_strike
            return self.max_pain
        except Exception:
            return None

    def calculate_pcr(self):
        try:
            total_put_oi = self.df["Put OI"].sum()
            total_call_oi = self.df["Call OI"].sum()
            self.pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 1.0
            return self.pcr
        except Exception:
            self.pcr = 1.0
            return self.pcr

    def detect_market_type(self):
        if self.spot_price is None or self.max_pain is None:
            return "UNKNOWN"
        distance_pct = abs(self.spot_price - self.max_pain) / self.spot_price * 100
        total_call_oi_change = abs(self.df["Call OI Change"].sum())
        total_put_oi_change = abs(self.df["Put OI Change"].sum())
        total_change = total_call_oi_change + total_put_oi_change
        oi_dominance = max(total_call_oi_change, total_put_oi_change) / total_change if total_change > 0 else 0.5
        if distance_pct > 0.35 and oi_dominance > 0.6:
            return "EXPANSION"
        elif distance_pct < 0.25 and 0.9 <= self.pcr <= 1.1:
            return "RANGE"
        else:
            return "NEUTRAL"

    def generate_signal(self, sbc_context=None):
        score = 0
        reasons = []
        direction = None
        current_hour = datetime.now().hour
        market_type = self.detect_market_type()

        if market_type == "RANGE":
            return {
                "signal": "AVOID",
                "score": 0,
                "confidence": 0,
                "market_type": market_type,
                "direction": None,
                "best_strike": None,
                "reasons": [
                    "✗ Market in RANGE/CHOP zone",
                    "✗ Theta decay will kill premium",
                    "✗ Wait for clear directional move",
                ],
                "action": "Stay out of the market",
                "base_score": 0,
                "sbc_score": 0,
            }

        if 11 <= current_hour < 15:
            score += 10
            reasons.append("✓ Trading in optimal time window (11 AM – 3 PM)")
        else:
            reasons.append("✗ Outside optimal trading hours")

        if self.spot_price and self.max_pain:
            if self.spot_price > self.max_pain:
                score += 20
                reasons.append(
                    f"✓ Price above Max Pain (Spot: {self.spot_price:.0f} > MP: {self.max_pain:.0f})"
                )
                direction = "CALLS"
            else:
                score += 15
                reasons.append(
                    f"✓ Price below Max Pain (Spot: {self.spot_price:.0f} < MP: {self.max_pain:.0f})"
                )
                direction = "PUTS"

        total_call_oi_change = self.df["Call OI Change"].sum()
        total_put_oi_change = self.df["Put OI Change"].sum()

        if direction == "CALLS" and total_put_oi_change < 0:
            score += 25
            reasons.append("✓ PUT OI unwinding (strong bullish signal)")
        elif direction == "PUTS" and total_call_oi_change < 0:
            score += 25
            reasons.append("✓ CALL OI unwinding (strong bearish signal)")

        if direction == "CALLS" and total_call_oi_change > 0:
            score += 20
            reasons.append("✓ Fresh CALL OI addition (new longs entering)")
        elif direction == "PUTS" and total_put_oi_change > 0:
            score += 20
            reasons.append("✓ Fresh PUT OI addition (new shorts entering)")

        if self.spot_price and self.max_pain:
            distance_pct = abs(self.spot_price - self.max_pain) / self.spot_price * 100
            if distance_pct > 0.35:
                score += 15
                reasons.append(
                    f"✓ Good distance from Max Pain ({distance_pct:.2f}% – good premium potential)"
                )
            else:
                score += 5

        base_score = score
        sbc_score = 0
        if sbc_context is not None:
            sbc_score, sbc_reasons = compute_sbc_score(sbc_context)
            score += sbc_score
            reasons.extend(sbc_reasons)

        if sbc_context and sbc_context.get("tithi_name") == "Rikta":
            if score >= 75:
                reasons.append("⚠️ Rikta tithi – cap conviction for safety")
                score = max(score, 70)

        if score >= 75:
            signal_type = "BUY"
            confidence = min(95, score)
        elif score >= 50:
            signal_type = "WAIT"
            confidence = score
        else:
            signal_type = "AVOID"
            confidence = 100 - score

        return {
            "signal": signal_type,
            "score": score,
            "confidence": confidence,
            "market_type": market_type,
            "direction": direction,
            "best_strike": None,
            "reasons": reasons,
            "action": self.get_action_message(signal_type),
            "base_score": base_score,
            "sbc_score": sbc_score,
        }

    def get_action_message(self, signal, direction=None):
        if signal == "BUY":
            return "🎯 BUY with strict stop-loss and proper position sizing"
        elif signal == "WAIT":
            return "⏳ Wait for clear price confirmation before entering"
        else:
            return "❌ No trade setup – Protect capital, stay in cash"

    def get_support_resistance(self):
        if self.df.empty:
            return None, None
        put_oi_max_idx = self.df["Put OI"].idxmax()
        call_oi_max_idx = self.df["Call OI"].idxmax()
        support = self.df.loc[put_oi_max_idx, "Strike Price"]
        resistance = self.df.loc[call_oi_max_idx, "Strike Price"]
        return support, resistance

# ============================================================================
# SMARTAPI CONNECTOR (CANDLES)
# ============================================================================

TOKEN_MAP = {
    "BANKNIFTY": "26009",
    "NIFTY": "26000",
}

def create_connection(api_key: str, client_code: str, password: str, totp: str):
    obj = SmartConnect(api_key=api_key)
    session = obj.generateSession(client_code, password, totp)
    if not session.get("status"):
        raise RuntimeError(f"SmartAPI login failed: {session}")
    return obj

def fetch_intraday_candles(conn, underlying: str, interval: str, lookback_minutes: int) -> pd.DataFrame:
    token = TOKEN_MAP[underlying]
    to_dt = datetime.now()
    from_dt = to_dt - timedelta(minutes=lookback_minutes)
    params = {
        "exchange": "NSE",
        "symboltoken": token,
        "interval": interval,
        "fromdate": from_dt.strftime("%Y-%m-%d %H:%M"),
        "todate": to_dt.strftime("%Y-%m-%d %H:%M"),
    }
    resp = conn.getCandleData(params)
    if "data" not in resp or resp["data"] is None:
        raise RuntimeError(f"No candle data returned: {resp}")
    data = resp["data"]
    cols = ["datetime", "open", "high", "low", "close", "volume"]
    df = pd.DataFrame(data, columns=cols)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna()

# ============================================================================
# INDICATORS (5 PRO INDICATORS)
# ============================================================================

def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    v = df["volume"].replace(0, np.nan)
    df["vwap"] = (tp * v).cumsum() / v.cumsum()
    df["vwap_deviation_pct"] = (df["close"] - df["vwap"]) / df["vwap"] * 100
    return df

def add_volume_delta(df: pd.DataFrame) -> pd.DataFrame:
    change = df["close"].diff()
    sign = np.where(change > 0, 1, np.where(change < 0, -1, 0))
    df["volume_delta"] = df["volume"] * sign
    df["cum_volume_delta"] = df["volume_delta"].cumsum()
    return df

def add_rsi_and_divergence(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    rsi = RSIIndicator(close=df["close"], window=window).rsi()
    df["rsi"] = rsi
    df["rsi_divergence"] = 0
    lookback = 20
    if len(df) >= window + lookback:
        recent = df.iloc[-lookback:]
        price_high_idx = recent["close"].idxmax()
        price_low_idx = recent["close"].idxmin()
        rsi_at_high = recent.loc[price_high_idx, "rsi"]
        rsi_before_high = recent["rsi"][:price_high_idx].max()
        if rsi_at_high < rsi_before_high:
            df.loc[price_high_idx, "rsi_divergence"] = -1
        rsi_at_low = recent.loc[price_low_idx, "rsi"]
        rsi_before_low = recent["rsi"][:price_low_idx].min()
        if rsi_at_low > rsi_before_low:
            df.loc[price_low_idx, "rsi_divergence"] = 1
    return df

def compute_volume_profile(df: pd.DataFrame, bins: int = 20) -> pd.Series:
    price_bins = pd.cut(df["close"], bins=bins)
    profile = df.groupby(price_bins)["volume"].sum()
    return profile

def compute_advance_decline(df: pd.DataFrame):
    advances = (df["close"] > df["open"]).sum()
    declines = (df["close"] < df["open"]).sum()
    total = advances + declines
    ad_ratio = advances / total if total > 0 else 0.5
    return {
        "advances": int(advances),
        "declines": int(declines),
        "ad_value": int(advances - declines),
        "ad_ratio": float(ad_ratio),
    }

def build_indicator_scores(df: pd.DataFrame):
    latest = df.iloc[-1]
    score = 0
    reasons = []

    if latest["vwap_deviation_pct"] > 0.1:
        score += 8
        reasons.append("✓ Price holding above VWAP (bullish bias)")
    elif latest["vwap_deviation_pct"] < -0.1:
        score -= 8
        reasons.append("✗ Price below VWAP (bearish bias)")

    recent_delta = df["volume_delta"].tail(min(10, len(df))).sum()
    if recent_delta > 0:
        score += 10
        reasons.append("✓ Positive volume delta (buyers more aggressive)")
    elif recent_delta < 0:
        score -= 10
        reasons.append("✗ Negative volume delta (sellers more aggressive)")

    if latest["rsi"] < 35:
        score += 6
        reasons.append("✓ RSI oversold (potential bounce)")
    elif latest["rsi"] > 65:
        score -= 6
        reasons.append("✗ RSI overbought (risk of exhaustion)")

    if latest.get("rsi_divergence", 0) == 1:
        score += 12
        reasons.append("★ Bullish RSI divergence (reversal up)")
    elif latest.get("rsi_divergence", 0) == -1:
        score -= 12
        reasons.append("★ Bearish RSI divergence (reversal down)")

    ad = compute_advance_decline(df)
    if ad["ad_ratio"] > 0.6:
        score += 8
        reasons.append("✓ Strong breadth (advances dominating)")
    elif ad["ad_ratio"] < 0.4:
        score -= 8
        reasons.append("✗ Weak breadth (declines dominating)")

    score = max(-60, min(60, score))
    return {"tech_score": score, "tech_reasons": reasons, "ad_stats": ad}

# ============================================================================
# COMBINED SCORING ENGINE
# ============================================================================

def combine_scores(oi_score, tech_score, sbc_score, market_type, tithi_name, mode="Scalping"):
    oi_score = max(0, min(80, oi_score))
    tech_score = max(-60, min(60, tech_score))
    sbc_score = max(-25, min(25, sbc_score))

    if mode == "Scalping":
        w_oi, w_tech, w_sbc = 0.35, 0.50, 0.15
        mode_bias = 0
    elif mode == "Intraday":
        w_oi, w_tech, w_sbc = 0.45, 0.40, 0.15
        mode_bias = 5
    else:
        w_oi, w_tech, w_sbc = 0.35, 0.25, 0.40
        mode_bias = 0

    blended = (
        w_oi * (oi_score - 40)
        + w_tech * tech_score
        + w_sbc * sbc_score
        + mode_bias
    )

    if market_type == "RANGE":
        blended = min(blended, 15)
    elif market_type == "EXPANSION":
        blended += 5

    if tithi_name == "Rikta":
        blended = min(blended, 10)
    elif tithi_name == "Nanda":
        blended += 5

    final_score = max(0, min(100, 50 + blended))

    if final_score >= 78:
        signal = "BUY"
    elif final_score >= 65:
        signal = "SCALP"
    elif final_score >= 52:
        signal = "WAIT"
    else:
        signal = "AVOID"

    confidence = 50 + abs(final_score - 50) * 0.7
    confidence = max(55, min(98, confidence))
    return signal, round(final_score), round(confidence)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.title("🌀 AstroQuant Pro V10.0 - SmartAPI Edition")
    st.markdown("**Real-Time Option Chain + Price/Volume + Vedic Astrology (SBC)**")

    with st.sidebar:
        st.header("⚙️ Configuration")
        mode = st.radio("Trading Mode", ["Scalping", "Intraday", "Swing"])
        data_mode = st.radio("Data Mode", ["CSV Only", "CSV + SmartAPI (Live)"])
        underlying = st.selectbox("Underlying", ["BANKNIFTY", "NIFTY"])
        timeframe = st.selectbox(
            "Spot timeframe",
            ["ONE_MINUTE", "FIVE_MINUTE", "FIFTEEN_MINUTE"],
            index=1,
        )
        lookback = st.slider("Lookback (minutes)", 30, 360, 120, 30)

        st.markdown("---")
        st.subheader("SmartAPI Login (daily OTP)")
        api_key = st.text_input("API Key", type="password")
        client_code = st.text_input("Client Code", type="password")
        password = st.text_input("Password", type="password")
        st.caption("Login to Angel, then read 6-digit TOTP from mobile app.")
        totp = st.text_input("TOTP", type="password")

        st.markdown("---")
        st.subheader("Option Chain CSV")
        uploaded_file = st.file_uploader(
            "Clean option-chain CSV",
            type=["csv"],
            help="Strike Price, Call OI, Call OI Change, Put OI, Put OI Change",
        )

        run_btn = st.button("🚀 RUN ANALYSIS", use_container_width=True)

    if not run_btn:
        st.info("Upload CSV, fill SmartAPI credentials (if using live), then click RUN ANALYSIS.")
        return
    if uploaded_file is None:
        st.error("Please upload a clean option-chain CSV.")
        return

    df_oc = pd.read_csv(uploaded_file)
    for col in ["Strike Price", "Call OI", "Call OI Change", "Put OI", "Put OI Change"]:
        if col not in df_oc.columns:
            st.error(f"CSV missing required column: {col}")
            return

    for col in ["Strike Price", "Call OI", "Call OI Change", "Put OI", "Put OI Change"]:
        df_oc[col] = pd.to_numeric(df_oc[col], errors="coerce")
    df_oc = df_oc.dropna(subset=["Strike Price"])
    df_oc = df_oc.sort_values("Strike Price").reset_index(drop=True)

    analyzer = OptionChainAnalyzer(df_oc)
    spot_price = float(df_oc["Strike Price"].median())
    spot_price = st.number_input("Spot Price (override if needed)", value=spot_price, step=100.0)
    analyzer.spot_price = spot_price
    analyzer.calculate_max_pain()
    analyzer.calculate_pcr()
    market_type = analyzer.detect_market_type()

    index_nakshatra = get_market_nakshatra(spot_price)
    moon_nakshatra = index_nakshatra
    tithi_name, tithi_num = get_tithi_info()
    sbc_context = {
        "moon_nakshatra": moon_nakshatra,
        "index_nakshatra": index_nakshatra,
        "tithi_name": tithi_name,
        "weekday": datetime.now().weekday(),
        "planetary_vedhas": [],
    }

    oc_signal = analyzer.generate_signal(sbc_context=sbc_context)
    oi_score = oc_signal["base_score"]
    sbc_score = oc_signal["sbc_score"]

    candles = None
    tech_block = {"tech_score": 0, "tech_reasons": [], "ad_stats": None}
    if data_mode == "CSV + SmartAPI (Live)":
        try:
            conn = create_connection(api_key, client_code, password, totp)
            candles = fetch_intraday_candles(conn, underlying, timeframe, lookback)
            candles = add_vwap(candles)
            candles = add_volume_delta(candles)
            candles = add_rsi_and_divergence(candles)
            tech_block = build_indicator_scores(candles)
        except Exception as e:
            st.error(f"SmartAPI / indicator error: {e}")

    tech_score = tech_block["tech_score"]
    final_signal, final_score, final_conf = combine_scores(
        oi_score=oi_score,
        tech_score=tech_score,
        sbc_score=sbc_score,
        market_type=market_type,
        tithi_name=tithi_name,
        mode=mode,
    )

    st.markdown("---")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("Spot", f"{spot_price:,.0f}")
    with c2:
        st.metric("Max Pain", f"{analyzer.max_pain:,.0f}" if analyzer.max_pain else "N/A")
    with c3:
        st.metric("PCR", f"{analyzer.pcr:.2f}")
    with c4:
        st.metric("Market Type", market_type)
    with c5:
        st.metric("Mode", mode.upper())
    with c6:
        st.metric("Strikes", len(df_oc))
    st.markdown("---")

    tab_signal, tab_structure, tab_oi, tab_pv, tab_sbc, tab_pro = st.tabs(
        ["🎯 Signal", "📊 Structure", "📈 OI", "📉 Price & Volume", "🌀 SBC (V10)", "💡 Pro Strategy"]
    )

    with tab_signal:
        st.subheader("🎯 Final Trading Signal")
        signal_colors = {
            "BUY": ("🟢", "buy-signal"),
            "SCALP": ("🟢", "buy-signal"),
            "WAIT": ("🟡", "wait-signal"),
            "AVOID": ("🔴", "avoid-signal"),
        }
        icon, _ = signal_colors[final_signal]
        s1, s2, s3 = st.columns(3)
        with s1:
            st.metric("Signal", f"{icon} {final_signal}")
        with s2:
            st.metric("Score", f"{final_score}/100")
        with s3:
            st.metric("Confidence", f"{final_conf}%")

        st.markdown("---")
        st.subheader("Score Breakdown")
        b1, b2, b3 = st.columns(3)
        with b1:
            st.metric("OI Base Score", f"{oi_score:.0f}")
        with b2:
            st.metric("Tech Score", f"{tech_score:.0f}")
        with b3:
            st.metric("SBC Score", f"{sbc_score:.0f}")

        st.markdown("---")
        st.subheader("Key Reasons (OI + SBC + Technical)")
        for r in oc_signal["reasons"]:
            st.write("•", r)
        for r in tech_block["tech_reasons"]:
            st.write("•", r)

    with tab_structure:
        st.subheader("📊 Market Structure")
        support, resistance = analyzer.get_support_resistance()
        cs1, cs2 = st.columns(2)
        with cs1:
            if support is not None and not (isinstance(support, float) and math.isnan(support)):
                st.metric("Support (Put OI Max)", f"{support:,.0f}")
            else:
                st.metric("Support (Put OI Max)", "N/A")
        with cs2:
            if resistance is not None and not (isinstance(resistance, float) and math.isnan(resistance)):
                st.metric("Resistance (Call OI Max)", f"{resistance:,.0f}")
            else:
                st.metric("Resistance (Call OI Max)", "N/A")

        st.markdown("---")
        fig = go.Figure()
        fig.add_bar(x=df_oc["Strike Price"], y=df_oc["Call OI"], name="Call OI", marker_color="indianred")
        fig.add_bar(x=df_oc["Strike Price"], y=df_oc["Put OI"], name="Put OI", marker_color="seagreen")
        fig.update_layout(barmode="group", height=400, xaxis_title="Strike", yaxis_title="Open Interest")
        st.plotly_chart(fig, use_container_width=True)

        if candles is not None:
            st.markdown("---")
            st.write("**Volume Profile (spot close)**")
            profile = compute_volume_profile(candles, bins=20)
            fig2 = go.Figure()
            fig2.add_bar(y=[f"{i.left:.0f}-{i.right:.0f}" for i in profile.index], x=profile.values, orientation="h", marker_color="steelblue")
            fig2.update_layout(height=500, xaxis_title="Volume", yaxis_title="Price Range")
            st.plotly_chart(fig2, use_container_width=True)

    with tab_oi:
        st.subheader("📈 Open Interest Analysis")
        o1, o2 = st.columns(2)
        with o1:
            st.metric("Total Call OI", f"{df_oc['Call OI'].sum():,.0f}")
        with o2:
            st.metric("Total Put OI", f"{df_oc['Put OI'].sum():,.0f}")

        fig3 = go.Figure()
        fig3.add_bar(x=df_oc["Strike Price"], y=df_oc["Call OI Change"], name="Call OI Change", marker_color="salmon")
        fig3.add_bar(x=df_oc["Strike Price"], y=df_oc["Put OI Change"], name="Put OI Change", marker_color="lightgreen")
        fig3.update_layout(barmode="group", height=400, xaxis_title="Strike", yaxis_title="OI Change")
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("---")
        st.dataframe(df_oc, use_container_width=True, height=400)

    with tab_pv:
        st.subheader("📉 Price, VWAP, Volume & RSI")
        if candles is None:
            st.info("Connect SmartAPI (CSV + SmartAPI mode) to see price/volume charts.")
        else:
            fig_p = go.Figure()
            fig_p.add_candlestick(x=candles.index, open=candles["open"], high=candles["high"], low=candles["low"], close=candles["close"], name="Price")
            fig_p.add_scatter(x=candles.index, y=candles["vwap"], name="VWAP", line=dict(color="orange", width=2))
            fig_p.update_layout(height=450, xaxis_title="Time", yaxis_title="Price")
            st.plotly_chart(fig_p, use_container_width=True)

            fig_v = go.Figure()
            fig_v.add_bar(x=candles.index, y=candles["volume"], name="Volume", marker_color="lightgrey")
            fig_v.add_bar(x=candles.index, y=candles["volume_delta"], name="Volume Delta", marker_color="teal")
            fig_v.update_layout(barmode="overlay", height=300, xaxis_title="Time", yaxis_title="Volume / Delta")
            st.plotly_chart(fig_v, use_container_width=True)

            fig_r = go.Figure()
            fig_r.add_scatter(x=candles.index, y=candles["rsi"], name="RSI", line=dict(color="purple"))
            fig_r.add_hline(y=30, line_dash="dash", line_color="grey")
            fig_r.add_hline(y=70, line_dash="dash", line_color="grey")

            bull_idx = candles.index[candles["rsi_divergence"] == 1]
            bear_idx = candles.index[candles["rsi_divergence"] == -1]
            if len(bull_idx) > 0:
                fig_r.add_scatter(x=bull_idx, y=candles.loc[bull_idx, "rsi"], mode="markers", marker=dict(color="green", size=9), name="Bullish Div")
            if len(bear_idx) > 0:
                fig_r.add_scatter(x=bear_idx, y=candles.loc[bear_idx, "rsi"], mode="markers", marker=dict(color="red", size=9), name="Bearish Div")
            fig_r.update_layout(height=300, xaxis_title="Time", yaxis_title="RSI")
            st.plotly_chart(fig_r, use_container_width=True)

    with tab_sbc:
        st.subheader("🌀 Surya-Brahma-Chandra (SBC) Analysis – V10")
        s1, s2, s3 = st.columns(3)
        with s1:
            st.metric("Moon Nakshatra", moon_nakshatra)
        with s2:
            st.metric("Index Nakshatra", index_nakshatra)
        with s3:
            st.metric("Tithi", f"{tithi_name} ({tithi_num})")

        st.markdown("---")
        st.write("**SBC Signal Components:**")
        st.write(f"• SBC Score: {sbc_score:.0f}")
        for r in [x for x in oc_signal["reasons"] if "tithi" in x.lower() or "nakshatra" in x.lower()]:
            st.write("•", r)
        st.caption("Swing mode gives higher weight to SBC score.")

    with tab_pro:
        st.subheader("💡 Pro Strategy Playbook")
        st.write(f"**Final Signal:** {final_signal} (Score {final_score}/100, {final_conf}% confidence)")
        st.write(f"**Mode:** {mode.upper()} | **Market Type:** {market_type} | **Tithi:** {tithi_name}")

        st.markdown("---")
        st.write("### Entry Rules")
        if final_signal in ("BUY", "SCALP"):
            st.write("• Enter in direction of signal when price respects VWAP and OI also supports move.")
            st.write("• Avoid fresh entries in RANGE market type or on Rikta tithi.")
        elif final_signal == "WAIT":
            st.write("• Wait for either VWAP breakout with positive volume delta or clear OI breakout.")
        else:
            st.write("• Stay in cash. Preserve capital until structure and indicators align.")

        st.write("### Stop-Loss & Targets")
        if mode == "Scalping":
            st.write("• Stop-loss: 0.3–0.5% on spot / 30–60 points on BANKNIFTY.")
            st.write("• Target: 1–1.5× risk, exit quickly on RSI extremes.")
        elif mode == "Intraday":
            st.write("• Stop-loss: below last swing low/high or key VWAP band.")
            st.write("• Target: 2× risk; trail using VWAP and OI shifts.")
        else:
            st.write("• Use higher timeframe levels; hold across sessions only when SBC + tech strongly aligned.")
            st.write("• Re-evaluate on new tithi / major gap openings.")

if __name__ == "__main__":
    main()
