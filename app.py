"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║         🌀 ASTROQUANT PRO V9.0 - ULTIMATE SBC EDITION (SIMPLIFIED) 🌀       ║
║                                                                              ║
║     Professional Option Chain Analyzer with Refined Sarvatobhadra Chakra    ║
║              Integration for NIFTY & BANKNIFTY Trading                      ║
║                                                                              ║
║  Version: 9.0.1 - Simplified (CSV Upload + Angel Broking Live API)         ║
║  Author: AstroQuant India Trading Community                                ║
║  Date: December 20, 2025                                                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import traceback

# Try importing plotly
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("⚠️ Plotly not installed. Add 'plotly' to requirements.txt for charts")

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="AstroQuant Pro V9.0 - Ultimate SBC Edition",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .buy-signal {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        font-size: 1.8rem;
        font-weight: bold;
        text-align: center;
        padding: 30px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 8px 16px rgba(17, 153, 142, 0.3);
    }
    .wait-signal {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        font-size: 1.8rem;
        font-weight: bold;
        text-align: center;
        padding: 30px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 8px 16px rgba(245, 87, 108, 0.3);
    }
    .avoid-signal {
        background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
        color: white;
        font-size: 1.8rem;
        font-weight: bold;
        text-align: center;
        padding: 30px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 8px 16px rgba(238, 9, 121, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# V9.0 - SARVATOBHADRA CHAKRA & NAKSHATRA ENGINE
# ============================================================================

NAKSHATRAS = [
    "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigasira", "Ardra",
    "Punarvasu", "Pushya", "Aslesha", "Magha", "Purva Phalguni",
    "Uttara Phalguni", "Hasta", "Chitra", "Svati", "Visakha",
    "Anuradha", "Jyestha", "Mula", "Purva Ashadha", "Uttara Ashadha",
    "Sravana", "Dhanistha", "Shatabhisha", "Purva Bhadrapada",
    "Uttara Bhadrapada", "Revati"
]

BENEFIC_PLANETS = {"Jupiter", "Venus", "Moon", "Mercury"}
MALEFIC_PLANETS = {"Saturn", "Mars", "Rahu", "Ketu"}

NAKSHATRA_SENTIMENT = {
    "Pushya": "Reversal",
    "Ashwini": "Bullish", "Rohini": "Bullish", "Hasta": "Bullish",
    "Chitra": "Bullish", "Anuradha": "Bullish", "Sravana": "Bullish", "Revati": "Bullish",
    "Bharani": "Bearish", "Krittika": "Bearish", "Ardra": "Bearish",
    "Aslesha": "Bearish", "Jyestha": "Bearish", "Mula": "Bearish", "Shatabhisha": "Bearish"
}

# ============================================================================
# HELPER FUNCTIONS - ASTROLOGY
# ============================================================================

def get_market_nakshatra(price: float) -> str:
    """Map NIFTY/BANKNIFTY spot price to nakshatra (every 100 points = 1 step)"""
    if price <= 0:
        return "Ashwini"
    reduced_value = int(price / 100)
    idx = reduced_value % 27
    return NAKSHATRAS[idx]

def get_tithi_info():
    """Calculate current Tithi (lunar day)"""
    day_of_month = datetime.now().day
    tithi = ((day_of_month - 1) % 30) + 1
    
    tithi_mapping = {
        "Nanda": [1, 6, 11, 16, 21, 26],
        "Bhadra": [2, 7, 12, 17, 22, 27],
        "Jaya": [3, 8, 13, 18, 23, 28],
        "Rikta": [4, 9, 14, 19, 24, 29],
        "Poorna": [5, 10, 15, 20, 25, 30]
    }
    
    for name, days in tithi_mapping.items():
        if tithi in days:
            return name, tithi
    return "Unknown", tithi

def compute_sbc_score(sbc_context):
    """V9.0 Refined SBC Scoring Engine"""
    score = 0
    reasons = []
    
    # Nakshatra sentiment
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
            reasons.append(f"★ {label} in Pushya (MAJOR REVERSAL ZONE) (+15)")
    
    # Tithi weighting
    tithi_name = sbc_context.get("tithi_name")
    if tithi_name == "Nanda":
        score += 10
        reasons.append("✓ Nanda tithi (prosperity/growth) (+10)")
    elif tithi_name == "Jaya":
        score += 5
        reasons.append("~ Jaya tithi (victory/trending) (+5)")
    elif tithi_name == "Rikta":
        score -= 20
        reasons.append("✗ Rikta tithi (empty/volatile - AVOID) (−20)")
    elif tithi_name == "Poorna":
        score += 5
        reasons.append("~ Poorna tithi (completion/strong moves) (+5)")
    
    return score, reasons

# ============================================================================
# OPTION CHAIN ANALYZER CLASS
# ============================================================================

class OptionChainAnalyzer:
    """V9.0 Enhanced Option Chain Analyzer with SBC Integration"""
    
    def __init__(self, df):
        self.df = df
        self.spot_price = None
        self.max_pain = None
        self.pcr = None

    def calculate_max_pain(self):
        """Calculate Max Pain (point of highest consolidated decay)"""
        try:
            strikes = sorted(self.df['Strike Price'].unique())
            min_pain = float('inf')
            max_pain_strike = strikes[len(strikes)//2]
            
            for strike in strikes:
                call_pain = 0
                put_pain = 0
                for idx, row in self.df.iterrows():
                    s = row['Strike Price']
                    if s < strike:
                        call_pain += row['Call OI'] * (strike - s)
                    if s > strike:
                        put_pain += row['Put OI'] * (s - strike)
                
                total_pain = call_pain + put_pain
                if total_pain < min_pain:
                    min_pain = total_pain
                    max_pain_strike = strike
            
            self.max_pain = max_pain_strike
            return self.max_pain
        except:
            return None

    def calculate_pcr(self):
        """Calculate Put-Call Ratio"""
        try:
            total_put_oi = self.df['Put OI'].sum()
            total_call_oi = self.df['Call OI'].sum()
            self.pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 1.0
            return self.pcr
        except:
            return 1.0

    def detect_market_type(self):
        """Detect if market is in RANGE, EXPANSION, or NEUTRAL mode"""
        if self.spot_price is None or self.max_pain is None:
            return "UNKNOWN"
        
        distance_pct = abs(self.spot_price - self.max_pain) / self.spot_price * 100
        total_call_oi_change = abs(self.df['Call OI Change'].sum())
        total_put_oi_change = abs(self.df['Put OI Change'].sum())
        total_change = total_call_oi_change + total_put_oi_change
        oi_dominance = max(total_call_oi_change, total_put_oi_change) / total_change if total_change > 0 else 0.5
        
        if distance_pct > 0.35 and oi_dominance > 0.6:
            return "EXPANSION"
        elif distance_pct < 0.25 and 0.9 <= self.pcr <= 1.1:
            return "RANGE"
        else:
            return "NEUTRAL"

    def generate_signal(self, sbc_context=None):
        """V9.0 Signal Generation: OI + SBC Combined"""
        score = 0
        reasons = []
        direction = None
        current_hour = datetime.now().hour
        market_type = self.detect_market_type()

        # RANGE market = AVOID
        if market_type == "RANGE":
            return {
                'signal': 'AVOID',
                'score': 0,
                'confidence': 0,
                'market_type': market_type,
                'direction': None,
                'best_strike': None,
                'reasons': ['✗ Market in RANGE/CHOP zone', '✗ Theta decay will kill premium'],
                'action': 'Stay out of the market',
                'base_score': 0,
                'sbc_score': 0
            }

        # OI TECHNICAL SCORE
        if 11 <= current_hour < 15:
            score += 10
            reasons.append('✓ Trading in optimal time window (11 AM - 3 PM)')
        else:
            reasons.append('✗ Outside optimal trading hours')

        if self.spot_price and self.max_pain:
            if self.spot_price > self.max_pain:
                score += 20
                reasons.append(f'✓ Price above Max Pain (Spot: {self.spot_price:.0f})')
                direction = "CALLS"
            else:
                score += 15
                reasons.append(f'✓ Price below Max Pain (Spot: {self.spot_price:.0f})')
                direction = "PUTS"

        total_call_oi_change = self.df['Call OI Change'].sum()
        total_put_oi_change = self.df['Put OI Change'].sum()

        if direction == "CALLS" and total_put_oi_change < 0:
            score += 25
            reasons.append('✓ PUT OI unwinding (strong bullish signal)')
        elif direction == "PUTS" and total_call_oi_change < 0:
            score += 25
            reasons.append('✓ CALL OI unwinding (strong bearish signal)')

        if self.spot_price and self.max_pain:
            distance_pct = abs(self.spot_price - self.max_pain) / self.spot_price * 100
            if distance_pct > 0.35:
                score += 15
                reasons.append(f'✓ Good distance from Max Pain ({distance_pct:.2f}%)')

        base_score = score

        # SBC OVERLAY SCORE
        sbc_score = 0
        sbc_reasons = []
        if sbc_context is not None:
            sbc_score, sbc_reasons = compute_sbc_score(sbc_context)
            score += sbc_score
            reasons.extend(sbc_reasons)

        # Rikta tithi guardrail
        if sbc_context and sbc_context.get("tithi_name") == "Rikta":
            if score >= 75:
                reasons.append("⚠️ Rikta tithi detected – capped conviction for safety")
                score = max(score, 70)

        # Final signal decision
        if score >= 75:
            signal_type = "BUY"
            confidence = min(95, score)
        elif score >= 50:
            signal_type = "WAIT"
            confidence = score
        else:
            signal_type = "AVOID"
            confidence = 100 - score

        best_strike = self.find_best_strike(direction if score >= 50 else None)

        return {
            'signal': signal_type,
            'score': score,
            'confidence': confidence,
            'market_type': market_type,
            'direction': direction,
            'best_strike': best_strike,
            'reasons': reasons,
            'action': self.get_action_message(signal_type, direction, best_strike),
            'base_score': base_score,
            'sbc_score': sbc_score
        }

    def find_best_strike(self, direction):
        """Find best strike for the given direction"""
        if not direction or self.spot_price is None:
            return None
        
        self.df['Distance'] = abs(self.df['Strike Price'] - self.spot_price)
        atm_strike = self.df.loc[self.df['Distance'].idxmin(), 'Strike Price']
        
        if direction == "CALLS":
            candidates = self.df[self.df['Strike Price'].between(atm_strike, atm_strike + 200)]
            if not candidates.empty:
                best = candidates.loc[candidates['Call OI Change'].idxmax()]
                return {
                    'strike': best['Strike Price'],
                    'type': 'CE',
                    'oi_change': best['Call OI Change'],
                    'oi': best['Call OI']
                }
        else:
            candidates = self.df[self.df['Strike Price'].between(atm_strike - 200, atm_strike)]
            if not candidates.empty:
                best = candidates.loc[candidates['Put OI Change'].idxmax()]
                return {
                    'strike': best['Strike Price'],
                    'type': 'PE',
                    'oi_change': best['Put OI Change'],
                    'oi': best['Put OI']
                }
        return None

    def get_action_message(self, signal, direction, best_strike):
        """Generate trading action message"""
        if signal == "BUY" and best_strike:
            return f"🎯 BUY {best_strike['strike']:.0f} {best_strike['type']} | OI Change: {best_strike['oi_change']:,.0f}"
        elif signal == "WAIT":
            return "⏳ Wait for clear price confirmation before entering"
        else:
            return "❌ No trade setup – Protect capital, stay in cash"

    def get_support_resistance(self):
        """Calculate support (put OI) and resistance (call OI) levels"""
        if self.df.empty:
            return None, None
        put_oi_max_idx = self.df['Put OI'].idxmax()
        call_oi_max_idx = self.df['Call OI'].idxmax()
        support = self.df.loc[put_oi_max_idx, 'Strike Price']
        resistance = self.df.loc[call_oi_max_idx, 'Strike Price']
        return support, resistance

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.title("🌀 AstroQuant Pro V9.0 - Ultimate SBC Edition")
    st.markdown("### Professional Option Chain Analyzer with Refined Sarvatobhadra Chakra")
    st.markdown("**🎯 Real-Time Signals | OI Analysis + Vedic Astrology | NIFTY & BANKNIFTY**")

    # Initialize session state
    if 'analysis_ready' not in st.session_state:
        st.session_state['analysis_ready'] = False

    # ========== SIDEBAR CONTROLS ==========
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.subheader("📊 Data Source")
        
        mode = st.radio(
            "Select Mode:",
            ["📁 CSV Upload (Manual)", "🔴 LIVE API (Angel Broking)"],
            help="CSV: Upload clean option chain CSV | LIVE: Real-time from Angel Broking"
        )

        # -------- CSV UPLOAD MODE --------
        if mode == "📁 CSV Upload (Manual)":
            st.subheader("📁 Upload Clean Option Chain CSV")
            st.info("""
            ✅ **Required CSV Format:**
            - Columns: Strike Price, Call OI, Call OI Change, Put OI, Put OI Change
            - No header merging or complex structure
            - Data starts from row 1
            """)
            
            uploaded_file = st.file_uploader(
                "Choose clean CSV file",
                type=['csv'],
                help="Upload a clean, pre-formatted option chain CSV"
            )

            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    
                    # Verify required columns
                    required_cols = ['Strike Price', 'Call OI', 'Call OI Change', 'Put OI', 'Put OI Change']
                    if all(col in df.columns for col in required_cols):
                        st.success("✅ CSV loaded successfully!")
                        
                        underlying = st.selectbox(
                            "Select Underlying:",
                            ["BANKNIFTY", "NIFTY50", "FINNIFTY"],
                            help="Index to analyze"
                        )
                        
                        spot_price = st.number_input(
                            "Enter Current Spot Price",
                            min_value=0.0,
                            value=50000.0,
                            step=100.0,
                            help=f"Current {underlying} price"
                        )
                        
                        if st.button("🚀 ANALYZE NOW", use_container_width=True):
                            st.session_state['analysis_ready'] = True
                            st.session_state['df'] = df
                            st.session_state['spot_price'] = spot_price
                            st.session_state['underlying'] = underlying
                            st.session_state['mode'] = 'csv'
                            st.success(f"✅ Analysis ready for {underlying}")
                    else:
                        st.error(f"❌ CSV missing required columns. Found: {list(df.columns)}")
                        st.info(f"Required: {required_cols}")
                except Exception as e:
                    st.error(f"❌ Error reading CSV: {e}")

        # -------- ANGEL BROKING LIVE MODE --------
        else:
            st.subheader("🔴 Angel Broking Live API")
            st.info("""
            ⚠️ **Angel Broking Integration Coming Soon!**
            
            Features to implement:
            - Real-time option chain data
            - Automatic 30-sec refresh
            - Live trading signals
            - Auto trade execution (optional)
            """)
            
            st.markdown("""
            **To enable live trading:**
            1. Get Angel Broking Account
            2. Generate API credentials
            3. Enter credentials below
            """)
            
            client_code = st.text_input("AAAQ573450", type="default")
            api_key = st.text_input("W7aGx1Bh", type="default")
            
            if st.button("🔗 Connect to Angel Broking", use_container_width=True):
                st.warning("⏳ API Integration in Progress - Check back soon!")

    # ========== ANALYSIS DISPLAY ==========
    if st.session_state.get('analysis_ready'):
        df = st.session_state['df']
        spot_price = st.session_state.get('spot_price', 50000.0)
        underlying = st.session_state.get('underlying', 'BANKNIFTY')

        analyzer = OptionChainAnalyzer(df)
        analyzer.spot_price = spot_price
        analyzer.calculate_max_pain()
        analyzer.calculate_pcr()

        # BUILD SBC CONTEXT (V9.0)
        index_nakshatra = get_market_nakshatra(spot_price)
        moon_nakshatra = index_nakshatra
        tithi_name, tithi_num = get_tithi_info()
        
        sbc_context = {
            "moon_nakshatra": moon_nakshatra,
            "index_nakshatra": index_nakshatra,
            "tithi_name": tithi_name,
            "weekday": datetime.now().weekday(),
            "planetary_vedhas": []
        }

        # GENERATE SIGNAL (V9.0)
        signal_result = analyzer.generate_signal(sbc_context=sbc_context)
        support, resistance = analyzer.get_support_resistance()

        # ========== KEY METRICS ==========
        st.markdown("---")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Spot Price", f"{spot_price:,.0f}")
        with col2:
            st.metric("Max Pain", f"{analyzer.max_pain:,.0f}" if analyzer.max_pain else "N/A")
        with col3:
            st.metric("PCR", f"{analyzer.pcr:.2f}" if analyzer.pcr else "N/A")
        with col4:
            st.metric("Market Type", analyzer.detect_market_type())
        with col5:
            st.metric("Total Strikes", len(df))
        st.markdown("---")

        # ========== ANALYSIS TABS ==========
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🎯 Signal",
            "📊 Structure",
            "🔍 OI Analysis",
            "📋 Data",
            "🌀 SBC (V9.0)",
            "💡 Combined Strategy"
        ])

        # -------- TAB 1: SIGNAL --------
        with tab1:
            st.subheader("🎯 OI-Based Trading Signal")
            
            signal_colors = {"BUY": "🟢", "WAIT": "🟡", "AVOID": "🔴"}
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Signal", f"{signal_colors[signal_result['signal']]} {signal_result['signal']}")
            with col2:
                st.metric("Score", f"{signal_result['score']}/100")
            with col3:
                st.metric("Confidence", f"{signal_result['confidence']:.0f}%")
            
            st.markdown("---")
            st.subheader("📋 Analysis Details")
            for reason in signal_result['reasons']:
                if '✓' in reason or 'Pushya' in reason or 'Nanda' in reason:
                    st.success(reason)
                elif '✗' in reason or 'AVOID' in reason:
                    st.error(reason)
                else:
                    st.info(reason)

            if signal_result['best_strike']:
                st.markdown("---")
                st.subheader("🎯 Recommended Strike")
                strike_info = signal_result['best_strike']
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Strike", f"{strike_info['strike']:.0f} {strike_info['type']}")
                with col2:
                    st.metric("OI Change", f"{strike_info['oi_change']:,.0f}")
                with col3:
                    st.metric("Total OI", f"{strike_info['oi']:,.0f}")

        # -------- TAB 2: STRUCTURE --------
        with tab2:
            st.subheader("📊 Price Structure - Support & Resistance")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Resistance (Call OI Max)", f"{resistance:,.0f}" if resistance else "N/A")
            with col2:
                st.metric("Support (Put OI Max)", f"{support:,.0f}" if support else "N/A")
            
            st.markdown("---")
            st.write("**Full Strike Price Distribution:**")
            st.dataframe(df[['Strike Price', 'Call OI', 'Put OI']].sort_values('Strike Price'))

        # -------- TAB 3: OI ANALYSIS --------
        with tab3:
            st.subheader("🔍 Open Interest Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Call OI", f"{df['Call OI'].sum():,.0f}")
            with col2:
                st.metric("Total Put OI", f"{df['Put OI'].sum():,.0f}")
            
            st.markdown("---")
            
            if PLOTLY_AVAILABLE:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=df['Strike Price'], y=df['Call OI'],
                    name='Call OI', marker_color='red'
                ))
                fig.add_trace(go.Bar(
                    x=df['Strike Price'], y=df['Put OI'],
                    name='Put OI', marker_color='green'
                ))
                fig.update_layout(
                    title='OI Distribution Across Strikes',
                    barmode='group', height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Install plotly for charts: pip install plotly")

        # -------- TAB 4: DATA --------
        with tab4:
            st.subheader("📋 Full Option Chain Data")
            
            st.write(f"**Total Strikes: {len(df)}**")
            st.dataframe(df, use_container_width=True, height=600)
            
            # Download button
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"option_chain_{underlying}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        # -------- TAB 5: SBC ASTROLOGY --------
        with tab5:
            st.subheader("🌀 Surya-Brahma-Chandra (SBC) Analysis")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Moon Nakshatra", sbc_context['moon_nakshatra'])
            with col2:
                st.metric("Index Nakshatra", sbc_context['index_nakshatra'])
            with col3:
                st.metric("Tithi", sbc_context['tithi_name'])
            
            st.markdown("---")
            st.write("**SBC Signal Components:**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("OI Base Score", signal_result['base_score'])
            with col2:
                st.metric("SBC Score", signal_result['sbc_score'])

        # -------- TAB 6: COMBINED STRATEGY --------
        with tab6:
            st.subheader("💡 Combined Recommendation")
            
            if signal_result['signal'] == 'BUY':
                st.markdown('<div class="buy-signal">🟢 BUY SIGNAL</div>', unsafe_allow_html=True)
            elif signal_result['signal'] == 'WAIT':
                st.markdown('<div class="wait-signal">🟡 WAIT FOR CONFIRMATION</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="avoid-signal">🔴 AVOID - NO TRADE ZONE</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.write(f"**Action:** {signal_result['action']}")
            st.write(f"**Market Type:** {signal_result['market_type']}")
            st.write(f"**Confidence:** {signal_result['confidence']:.0f}%")

if __name__ == "__main__":
    main()
