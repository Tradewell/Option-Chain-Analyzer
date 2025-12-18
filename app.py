import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Try importing plotly
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("⚠️ Plotly not installed. Charts will be limited. Add 'plotly' to requirements.txt")

# Page Configuration
st.set_page_config(
    page_title="Option Chain Analyzer - AstroQuant Pro",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .signal-buy { color: #00ff41; font-weight: bold; font-size: 24px; }
    .signal-wait { color: #ffd700; font-weight: bold; font-size: 24px; }
    .signal-avoid { color: #ff6b6b; font-weight: bold; font-size: 24px; }
    .metric-box { padding: 20px; border-radius: 10px; background-color: #f0f2f6; margin: 10px 0; }
    .info-box { padding: 15px; border-left: 4px solid #0088ff; background-color: #f0f7ff; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SIMPLIFIED CSV CONVERTER - Handles both formatted and raw NSE CSVs
# ============================================================================
def convert_csv_to_standard_format(raw_df):
    """
    Intelligent converter that detects CSV format and converts to standard format:
    Strike Price | Call OI | Call OI Change | Put OI | Put OI Change
    """
    try:
        with st.expander("🔄 CSV Conversion Details", expanded=False):
            st.write(f"**Total columns found:** {len(raw_df.columns)}")
            st.write(f"**Total rows:** {len(raw_df)}")
            st.write("**Column names:**")
            st.write(list(raw_df.columns))
        
        # ===== METHOD 1: Check if already in formatted format =====
        expected_cols = {'Strike Price', 'Call OI', 'Call OI Change', 'Put OI', 'Put OI Change'}
        actual_cols = set(raw_df.columns)
        
        if expected_cols.issubset(actual_cols):
            st.success("✅ CSV already in correct format!")
            clean_df = raw_df[['Strike Price', 'Call OI', 'Call OI Change', 'Put OI', 'Put OI Change']].copy()
            for col in clean_df.columns:
                clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')
            clean_df = clean_df.dropna(subset=['Strike Price'])
            return clean_df
        
        # ===== METHOD 2: Detect and convert raw NSE format =====
        st.info("Converting from raw NSE format...")
        
        if len(raw_df.columns) >= 5:
            strike_col_idx = None
            
            for i, col in enumerate(raw_df.columns):
                if 'strike' in str(col).lower():
                    strike_col_idx = i
                    break
            
            if strike_col_idx is None:
                for i, col in enumerate(raw_df.columns):
                    try:
                        numeric_vals = pd.to_numeric(raw_df[col], errors='coerce').dropna()
                        if len(numeric_vals) > 0:
                            if 40000 < numeric_vals.median() < 60000:
                                strike_col_idx = i
                                break
                    except:
                        continue
            
            if strike_col_idx is None:
                strike_col_idx = len(raw_df.columns) // 2
            
            st.success(f"✓ Strike Price at column {strike_col_idx + 1}")
            
            clean_df = pd.DataFrame()
            clean_df['Strike Price'] = pd.to_numeric(raw_df.iloc[:, strike_col_idx], errors='coerce')
            
            if strike_col_idx >= 2:
                clean_df['Call OI'] = pd.to_numeric(raw_df.iloc[:, strike_col_idx - 2], errors='coerce').fillna(0)
                clean_df['Call OI Change'] = pd.to_numeric(raw_df.iloc[:, strike_col_idx - 1], errors='coerce').fillna(0)
            else:
                clean_df['Call OI'] = 0
                clean_df['Call OI Change'] = 0
            
            if strike_col_idx + 2 < len(raw_df.columns):
                clean_df['Put OI'] = pd.to_numeric(raw_df.iloc[:, strike_col_idx + 1], errors='coerce').fillna(0)
                clean_df['Put OI Change'] = pd.to_numeric(raw_df.iloc[:, strike_col_idx + 2], errors='coerce').fillna(0)
            else:
                clean_df['Put OI'] = 0
                clean_df['Put OI Change'] = 0
            
            st.success(f"✅ Columns mapped successfully")
        else:
            st.error("❌ CSV does not have enough columns")
            return None
        
        clean_df = clean_df.dropna(subset=['Strike Price'])
        clean_df = clean_df[(clean_df['Call OI'] > 0) | (clean_df['Put OI'] > 0)]
        clean_df = clean_df.sort_values('Strike Price').reset_index(drop=True)
        
        if len(clean_df) == 0:
            st.error("❌ No valid data found after conversion")
            return None
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Strikes", len(clean_df))
        with col2:
            st.metric("Call OI", f"{clean_df['Call OI'].sum():,.0f}")
        with col3:
            st.metric("Put OI", f"{clean_df['Put OI'].sum():,.0f}")
        with col4:
            pcr = clean_df['Put OI'].sum() / clean_df['Call OI'].sum() if clean_df['Call OI'].sum() > 0 else 0
            st.metric("PCR", f"{pcr:.2f}")
        
        return clean_df
        
    except Exception as e:
        st.error(f"❌ Conversion error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

# ============================================================================
# SARVATOBHADRA CHAKRA MODULE
# ============================================================================

NAKSHATRAS = [
    "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigasira",
    "Ardra", "Punarvasu", "Pushya", "Aslesha", "Magha",
    "Purva Phalguni", "Uttara Phalguni", "Hasta", "Chitra", "Svati",
    "Visakha", "Anuradha", "Jyestha", "Mula", "Purva Ashadha",
    "Uttara Ashadha", "Sravana", "Dhanistha", "Shatabhisha",
    "Purva Bhadrapada", "Uttara Bhadrapada", "Revati", "Abhijit"
]

BENIGN_NAKSHATRAS = {"Pushya", "Hasta", "Chitra", "Anuradha", "Sravana", "Revati", "Ashwini", "Magha", "Uttara Phalguni"}
MALEFIC_NAKSHATRAS = {"Krittika", "Ardra", "Aslesha", "Jyestha", "Mula", "Shatabhisha", "Bharani", "Svati"}

def get_market_nakshatra(price):
    """Convert market price to nakshatra"""
    reduced_value = int(price / 100)
    nakshatra_idx = reduced_value % 28
    return NAKSHATRAS[nakshatra_idx]

def get_tithi_info():
    """Get tithi based on lunar calendar"""
    day_of_month = datetime.now().day
    tithi = ((day_of_month - 1) % 30) + 1
    
    tithi_types = {
        'Nanda': [1, 6, 11, 16, 21, 26],
        'Bhadra': [2, 7, 12, 17, 22, 27],
        'Jaya': [3, 8, 13, 18, 23, 28],
        'Rikta': [4, 9, 14, 19, 24, 29],
        'Poorna': [5, 10, 15, 20, 25, 30]
    }
    
    for name, tithis in tithi_types.items():
        if tithi in tithis:
            return name, tithi
    return "Unknown", tithi

def get_sbc_signal(spot_price, max_pain, signal_result):
    """Generate SBC-based signal"""
    market_nak = get_market_nakshatra(spot_price)
    max_pain_nak = get_market_nakshatra(max_pain)
    tithi_name, tithi_num = get_tithi_info()
    
    # Calculate vedha positions
    current_nak_idx = NAKSHATRAS.index(market_nak)
    support_idx = (current_nak_idx - 7) % 28
    resistance_idx = (current_nak_idx + 7) % 28
    
    support_nak = NAKSHATRAS[support_idx]
    resistance_nak = NAKSHATRAS[resistance_idx]
    
    # Determine combined signal
    oi_signal = signal_result['signal']
    
    if tithi_name == "Nanda" and oi_signal == "BUY":
        combined = "🟢 STRONG BUY"
        confidence_boost = 25
    elif tithi_name == "Rikta":
        combined = "🔴 AVOID (Rikta Phase)"
        confidence_boost = -40
    elif tithi_name == "Bhadra":
        combined = "🟡 RANGE-BOUND"
        confidence_boost = -15
    else:
        combined = "🟡 NEUTRAL"
        confidence_boost = 0
    
    final_confidence = min(95, max(0, signal_result['confidence'] + confidence_boost))
    
    return {
        'market_nakshatra': market_nak,
        'max_pain_nakshatra': max_pain_nak,
        'support_nakshatra': support_nak,
        'resistance_nakshatra': resistance_nak,
        'tithi_name': tithi_name,
        'tithi_num': tithi_num,
        'combined_signal': combined,
        'final_confidence': final_confidence,
        'nak_quality': 'Benign' if market_nak in BENIGN_NAKSHATRAS else ('Malefic' if market_nak in MALEFIC_NAKSHATRAS else 'Neutral')
    }

# ============================================================================
# SIGNAL ENGINE CLASS - ENHANCED
# ============================================================================
class OptionChainAnalyzer:
    def __init__(self, df):
        self.df = df
        self.spot_price = None
        self.max_pain = None
        self.pcr = None
    
    def calculate_max_pain(self):
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
        try:
            total_put_oi = self.df['Put OI'].sum()
            total_call_oi = self.df['Call OI'].sum()
            self.pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 1.0
            return self.pcr
        except:
            return 1.0
    
    def detect_market_type(self):
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
    
    def get_support_resistance(self):
        """Find support (high Put OI) and resistance (high Call OI) levels"""
        resistance = self.df.loc[self.df['Call OI'].idxmax(), 'Strike Price']
        support = self.df.loc[self.df['Put OI'].idxmax(), 'Strike Price']
        return support, resistance
    
    def get_oi_concentration(self):
        """Calculate OI concentration zones"""
        call_concentration = self.df.nlargest(3, 'Call OI')[['Strike Price', 'Call OI']]
        put_concentration = self.df.nlargest(3, 'Put OI')[['Strike Price', 'Put OI']]
        
        return call_concentration, put_concentration
    
    def calculate_oi_change_analysis(self):
        """Analyze buildup vs unwinding"""
        total_call_buildup = self.df[self.df['Call OI Change'] > 0]['Call OI Change'].sum()
        total_call_unwind = abs(self.df[self.df['Call OI Change'] < 0]['Call OI Change'].sum())
        
        total_put_buildup = self.df[self.df['Put OI Change'] > 0]['Put OI Change'].sum()
        total_put_unwind = abs(self.df[self.df['Put OI Change'] < 0]['Put OI Change'].sum())
        
        return {
            'call_buildup': total_call_buildup,
            'call_unwind': total_call_unwind,
            'put_buildup': total_put_buildup,
            'put_unwind': total_put_unwind
        }
    
    def get_pcr_interpretation(self):
        """PCR-based market interpretation"""
        if self.pcr < 0.8:
            return "🔴 BEARISH - Calls dominating, Puts weak"
        elif 0.8 <= self.pcr < 1.0:
            return "🟡 NEUTRAL-BEARISH - Slight call dominance"
        elif 1.0 <= self.pcr < 1.2:
            return "🟢 BALANCED - Neutral market structure"
        elif 1.2 <= self.pcr < 1.5:
            return "🟡 NEUTRAL-BULLISH - Slight put dominance"
        else:
            return "🔴 BULLISH - Puts dominating, Calls weak"
    
    def generate_signal(self):
        score = 0
        reasons = []
        direction = None
        current_hour = datetime.now().hour
        market_type = self.detect_market_type()
        
        if market_type == "RANGE":
            return {
                'signal': 'AVOID',
                'score': 0,
                'confidence': 0,
                'market_type': market_type,
                'direction': None,
                'best_strike': None,
                'reasons': [
                    '✗ Market in RANGE/CHOP zone',
                    '✗ Theta decay will kill premium',
                    '✗ Wait for clear directional move'
                ],
                'action': 'Stay out of the market'
            }
        
        if 11 <= current_hour < 15:
            score += 10
            reasons.append('✓ Trading in optimal time window')
        else:
            reasons.append('✗ Outside optimal trading hours')
        
        if self.spot_price and self.max_pain:
            if self.spot_price > self.max_pain:
                score += 20
                reasons.append(f'✓ Price above Max Pain (Spot: {self.spot_price:.0f} > MP: {self.max_pain:.0f})')
                direction = "CALLS"
            else:
                score += 15
                reasons.append(f'✓ Price below Max Pain (Spot: {self.spot_price:.0f} < MP: {self.max_pain:.0f})')
                direction = "PUTS"
        
        total_call_oi_change = self.df['Call OI Change'].sum()
        total_put_oi_change = self.df['Put OI Change'].sum()
        
        if direction == "CALLS" and total_put_oi_change < 0:
            score += 25
            reasons.append('✓ PUT OI unwinding (Bullish)')
        elif direction == "PUTS" and total_call_oi_change < 0:
            score += 25
            reasons.append('✓ CALL OI unwinding (Bearish)')
        
        if direction == "CALLS" and total_call_oi_change > 0:
            score += 20
            reasons.append('✓ Fresh CALL OI addition')
        elif direction == "PUTS" and total_put_oi_change > 0:
            score += 20
            reasons.append('✓ Fresh PUT OI addition')
        
        if self.spot_price and self.max_pain:
            distance_pct = abs(self.spot_price - self.max_pain) / self.spot_price * 100
            if distance_pct > 0.35:
                score += 15
                reasons.append(f'✓ Good distance from Max Pain ({distance_pct:.2f}%)')
            else:
                score += 5
        
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
            'action': self.get_action_message(signal_type, direction, best_strike)
        }
    
    def find_best_strike(self, direction):
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
        if signal == "BUY" and best_strike:
            return f"🎯 BUY {best_strike['strike']:.0f} {best_strike['type']} | OI Change: {best_strike['oi_change']:,.0f}"
        elif signal == "WAIT":
            return "⏳ Wait for clear price confirmation"
        else:
            return "❌ No trade - Protect capital"

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    st.title("🌀 AstroQuant Pro - Option Chain Analyzer")
    st.markdown("*Combining OI Analysis + Sarvatobhadra Chakra for Options Trading*")
    st.markdown("---")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("1️⃣ Upload CSV")
        uploaded_file = st.file_uploader(
            "Choose formatted NSE option chain CSV",
            type=['csv'],
            help="Use the CSV generated by convert_nse.py script"
        )
        
        if uploaded_file is not None:
            raw_df = pd.read_csv(uploaded_file)
            df = convert_csv_to_standard_format(raw_df)
            
            if df is not None:
                st.success("✅ CSV loaded successfully!")
                
                st.subheader("2️⃣ Market Data")
                spot_price = st.number_input(
                    "Enter Current Spot Price",
                    min_value=0.0,
                    value=50000.0,
                    step=50.0,
                    help="Current market price of the underlying"
                )
                
                expiry = st.text_input(
                    "Enter Expiry Date",
                    value="30-Dec-2025",
                    help="e.g., 30-Dec-2025"
                )
                
                if st.button("🚀 Analyze & Generate Signal", use_container_width=True):
                    st.session_state['analysis_ready'] = True
                    st.session_state['df'] = df
                    st.session_state['spot_price'] = spot_price
                    st.session_state['expiry'] = expiry

    if 'analysis_ready' in st.session_state and st.session_state['analysis_ready']:
        df = st.session_state['df']
        spot_price = st.session_state['spot_price']
        expiry = st.session_state['expiry']
        
        analyzer = OptionChainAnalyzer(df)
        analyzer.spot_price = spot_price
        analyzer.calculate_max_pain()
        analyzer.calculate_pcr()
        
        st.header(f"📈 Analysis: {expiry}")
        
        # ========== KEY METRICS ROW ==========
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Spot Price", f"{spot_price:,.0f}")
        with col2:
            st.metric("Max Pain", f"{analyzer.max_pain:,.0f}")
        with col3:
            st.metric("PCR", f"{analyzer.pcr:.2f}")
        with col4:
            st.metric("Market Type", analyzer.detect_market_type())
        with col5:
            st.metric("Total Strikes", len(df))
        
        st.markdown("---")
        
        # ========== ANALYSIS TABS ==========
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🎯 Signal", 
            "📊 Market Structure", 
            "🔍 OI Analysis", 
            "📋 Data",
            "🌀 Sarvatobhadra",
            "💡 Combined Strategy"
        ])
        
        # Generate signals
        signal_result = analyzer.generate_signal()
        sbc_signal = get_sbc_signal(spot_price, analyzer.max_pain, signal_result)
        
        with tab1:
            signal_colors = {"BUY": "🟢", "WAIT": "🟡", "AVOID": "🔴"}
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Signal", f"{signal_colors[signal_result['signal']]} {signal_result['signal']}")
            with col2:
                st.metric("Score", f"{signal_result['score']}/100")
            with col3:
                st.metric("Confidence", f"{signal_result['confidence']:.0f}%")
            
            st.subheader("📋 Analysis Details")
            for i, reason in enumerate(signal_result['reasons'], 1):
                st.write(f"{i}. {reason}")
            
            st.markdown("---")
            st.info(f"**Action:** {signal_result['action']}")
        
        with tab2:
            support, resistance = analyzer.get_support_resistance()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Support (Max Put OI)", f"{support:,.0f}", delta=f"{support - spot_price:.0f}")
            with col2:
                st.metric("Resistance (Max Call OI)", f"{resistance:,.0f}", delta=f"{resistance - spot_price:.0f}")
            
            st.markdown("---")
            st.subheader("📊 PCR Interpretation")
            st.info(analyzer.get_pcr_interpretation())
            
            st.markdown("---")
            st.subheader("💹 OI Buildup vs Unwinding")
            oi_analysis = analyzer.calculate_oi_change_analysis()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Call Buildup", f"{oi_analysis['call_buildup']:,.0f}")
            with col2:
                st.metric("Call Unwind", f"{oi_analysis['call_unwind']:,.0f}")
            with col3:
                st.metric("Put Buildup", f"{oi_analysis['put_buildup']:,.0f}")
            with col4:
                st.metric("Put Unwind", f"{oi_analysis['put_unwind']:,.0f}")
        
        with tab3:
            st.subheader("🎯 OI Concentration Zones")
            
            call_conc, put_conc = analyzer.get_oi_concentration()
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Top Call OI Strikes**")
                st.dataframe(call_conc, use_container_width=True, hide_index=True)
            with col2:
                st.write("**Top Put OI Strikes**")
                st.dataframe(put_conc, use_container_width=True, hide_index=True)
            
            if PLOTLY_AVAILABLE:
                st.markdown("---")
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=df['Strike Price'],
                    y=df['Call OI'],
                    name='Call OI',
                    marker_color='lightblue'
                ))
                fig.add_trace(go.Bar(
                    x=df['Strike Price'],
                    y=-df['Put OI'],
                    name='Put OI',
                    marker_color='lightcoral'
                ))
                
                fig.add_vline(x=spot_price, line_dash="dash", line_color="green", annotation_text="Spot Price")
                fig.add_vline(x=analyzer.max_pain, line_dash="dash", line_color="red", annotation_text="Max Pain")
                
                fig.update_layout(
                    title="Call OI vs Put OI Distribution",
                    xaxis_title="Strike Price",
                    yaxis_title="Open Interest",
                    barmode='relative',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("📊 Full Option Chain Data")
            st.dataframe(df, use_container_width=True)
            
            csv_buffer = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Analysis CSV",
                data=csv_buffer,
                file_name=f"analysis_{expiry.replace('-', '')}.csv",
                mime="text/csv"
            )
        
        with tab5:
            st.subheader("🌀 Sarvatobhadra Chakra Analysis")
            st.markdown("*Vedic Astrology Market Timing & Strike Selection*")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Market Nakshatra", sbc_signal['market_nakshatra'])
            with col2:
                st.metric("Max Pain Nakshatra", sbc_signal['max_pain_nakshatra'])
            with col3:
                st.metric("Tithi", f"{sbc_signal['tithi_name']} ({sbc_signal['tithi_num']})")
            with col4:
                weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                st.metric("Day (Vara)", weekday_names[datetime.now().weekday()])
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Nakshatra Quality:**")
                if sbc_signal['nak_quality'] == 'Benign':
                    st.success(f"✅ {sbc_signal['market_nakshatra']} is BENIGN")
                    st.write("- Favorable for buying")
                elif sbc_signal['nak_quality'] == 'Malefic':
                    st.error(f"⚠️ {sbc_signal['market_nakshatra']} is CHALLENGING")
                    st.write("- Caution advised")
                else:
                    st.info(f"~ {sbc_signal['market_nakshatra']} is NEUTRAL")
            
            with col2:
                st.write("**Tithi Phase:**")
                if sbc_signal['tithi_name'] == "Nanda":
                    st.success("✅ Growth Phase (Favorable)")
                elif sbc_signal['tithi_name'] == "Rikta":
                    st.error("⚠️ Loss Phase (AVOID)")
                else:
                    st.info(f"~ {sbc_signal['tithi_name']} Phase")
            
            st.markdown("---")
            
            st.subheader("🎯 SBC-Based Support/Resistance")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Support Nakshatra", sbc_signal['support_nakshatra'])
            
            with col2:
                st.metric("Resistance Nakshatra", sbc_signal['resistance_nakshatra'])
        
        with tab6:
            st.subheader("💡 Combined OI + SBC Strategy")
            st.markdown("*Best of Both Worlds: Technical OI Analysis + Vedic Astrology*")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**OI Signal:**")
                st.info(f"{signal_result['signal']} ({signal_result['confidence']:.0f}% confidence)")
            
            with col2:
                st.write("**SBC Signal:**")
                st.success(sbc_signal['combined_signal'])
            
            st.markdown("---")
            
            st.metric("Final Recommendation", sbc_signal['combined_signal'])
            st.metric("Final Confidence (Combined)", f"{sbc_signal['final_confidence']:.0f}%")
            
            st.markdown("---")
            
            st.success("""
            **🎯 TRADE SETUP RECOMMENDATION:**
            
            1. **Entry Signal:** Based on OI analysis (Tab 1)
            2. **Strike Selection:** Based on SBC nakshatras (Tab 5)
            3. **Timing:** Optimized by Tithi phase (Tab 5)
            4. **Confidence:** Combined OI + SBC verification
            5. **Stop Loss:** Support level from SBC
            6. **Target:** Resistance level from SBC
            
            ✅ Use this combined approach for higher probability trades!
            """)
    else:
        st.info("👈 Please upload a CSV file and click 'Analyze' to get started")

if __name__ == "__main__":
    main()
