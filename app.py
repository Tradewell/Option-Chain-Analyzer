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
    page_title="Option Chain Analyzer - Buyer's Edge",
    page_icon="📊",
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
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SIMPLIFIED CSV CONVERTER - Handles both formatted and raw NSE CSVs
# ============================================================================
def convert_csv_to_standard_format(raw_df):
    """
    Intelligent converter that detects CSV format and converts to standard format:
    Strike Price | Call OI | Call OI Change | Put OI | Put OI Change
    
    Handles:
    1. Pre-formatted CSV (from convert_nse.py script)
    2. Raw NSE CSV (CALLS/PUTS header format)
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
            # Just ensure proper column order and data types
            clean_df = raw_df[['Strike Price', 'Call OI', 'Call OI Change', 'Put OI', 'Put OI Change']].copy()
            for col in clean_df.columns:
                clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')
            clean_df = clean_df.dropna(subset=['Strike Price'])
            return clean_df
        
        # ===== METHOD 2: Detect and convert raw NSE format =====
        st.info("Converting from raw NSE format...")
        
        # Strategy: Look for columns with 5 numeric fields (typical NSE structure after header=1)
        # The columns should be: OI, Change OI (for calls), Strike, OI, Change OI (for puts)
        
        if len(raw_df.columns) >= 5:
            # Try to find strike column (usually in the middle or named "STRIKE")
            strike_col_idx = None
            
            # Check for "STRIKE" keyword first
            for i, col in enumerate(raw_df.columns):
                if 'strike' in str(col).lower():
                    strike_col_idx = i
                    break
            
            # If not found, look for column with values in strike price range
            if strike_col_idx is None:
                for i, col in enumerate(raw_df.columns):
                    try:
                        numeric_vals = pd.to_numeric(raw_df[col], errors='coerce').dropna()
                        if len(numeric_vals) > 0:
                            if 40000 < numeric_vals.median() < 60000:  # Typical strike range for Bank Nifty
                                strike_col_idx = i
                                break
                    except:
                        continue
            
            # If still not found, assume middle column is strike
            if strike_col_idx is None:
                strike_col_idx = len(raw_df.columns) // 2
            
            st.success(f"✓ Strike Price at column {strike_col_idx + 1}")
            
            # Build output dataframe using column positions
            # Assuming format: Call OI | Call Change | Strike | Put Change | Put OI
            # OR: Call OI | Call Change | Strike | Put OI | Put Change (check both patterns)
            
            clean_df = pd.DataFrame()
            clean_df['Strike Price'] = pd.to_numeric(raw_df.iloc[:, strike_col_idx], errors='coerce')
            
            # Try to assign Call OI and Call Change (usually 2 columns before strike)
            if strike_col_idx >= 2:
                clean_df['Call OI'] = pd.to_numeric(raw_df.iloc[:, strike_col_idx - 2], errors='coerce').fillna(0)
                clean_df['Call OI Change'] = pd.to_numeric(raw_df.iloc[:, strike_col_idx - 1], errors='coerce').fillna(0)
            else:
                clean_df['Call OI'] = 0
                clean_df['Call OI Change'] = 0
            
            # Try to assign Put OI and Put Change (usually after strike)
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
        
        # Clean data
        clean_df = clean_df.dropna(subset=['Strike Price'])
        clean_df = clean_df[(clean_df['Call OI'] > 0) | (clean_df['Put OI'] > 0)]
        clean_df = clean_df.sort_values('Strike Price').reset_index(drop=True)
        
        if len(clean_df) == 0:
            st.error("❌ No valid data found after conversion")
            return None
        
        # Show summary
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
# SIGNAL ENGINE CLASS
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
    
    def generate_signal(self):
        score = 0
        reasons = []
        direction = None
        current_hour = datetime.now().hour
        market_type = self.detect_market_type()
        
        # RANGE market - AVOID
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
        
        # Time Window Analysis
        if 11 <= current_hour < 15:
            score += 10
            reasons.append('✓ Trading in optimal time window')
        else:
            reasons.append('✗ Outside optimal trading hours')
        
        # Price vs Max Pain
        if self.spot_price and self.max_pain:
            if self.spot_price > self.max_pain:
                score += 20
                reasons.append(f'✓ Price above Max Pain (Spot: {self.spot_price:.0f} > MP: {self.max_pain:.0f})')
                direction = "CALLS"
            else:
                score += 15
                reasons.append(f'✓ Price below Max Pain (Spot: {self.spot_price:.0f} < MP: {self.max_pain:.0f})')
                direction = "PUTS"
        
        # OI Analysis
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
        
        # Max Pain Distance
        if self.spot_price and self.max_pain:
            distance_pct = abs(self.spot_price - self.max_pain) / self.spot_price * 100
            if distance_pct > 0.35:
                score += 15
                reasons.append(f'✓ Good distance from Max Pain ({distance_pct:.2f}%)')
            else:
                score += 5
        
        # Generate Signal
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
    st.title("📊 Option Chain Analyzer - Buyer's Edge")
    st.markdown("---")
    
    # Sidebar - Upload & Config
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # File Upload
        st.subheader("1️⃣ Upload CSV")
        uploaded_file = st.file_uploader(
            "Choose formatted NSE option chain CSV",
            type=['csv'],
            help="Use the CSV generated by convert_nse.py script"
        )
        
        if uploaded_file is not None:
            # Read the CSV
            raw_df = pd.read_csv(uploaded_file)
            
            # Convert to standard format
            df = convert_csv_to_standard_format(raw_df)
            
            if df is not None:
                st.success("✅ CSV loaded successfully!")
                
                # Spot Price Input
                st.subheader("2️⃣ Market Data")
                spot_price = st.number_input(
                    "Enter Current Spot Price",
                    min_value=0.0,
                    value=50000.0,
                    step=50.0,
                    help="Current market price of the underlying"
                )
                
                # Expiry Selection
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

    # Main Content Area
    if 'analysis_ready' in st.session_state and st.session_state['analysis_ready']:
        df = st.session_state['df']
        spot_price = st.session_state['spot_price']
        expiry = st.session_state['expiry']
        
        # Initialize Analyzer
        analyzer = OptionChainAnalyzer(df)
        analyzer.spot_price = spot_price
        analyzer.calculate_max_pain()
        analyzer.calculate_pcr()
        
        # Display Analysis Results
        st.header(f"📈 Analysis: {expiry}")
        
        # Key Metrics
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
        
        # Generate Signal
        signal_result = analyzer.generate_signal()
        
        # Display Signal
        st.subheader("🎯 Trading Signal")
        
        signal_colors = {
            "BUY": "🟢",
            "WAIT": "🟡",
            "AVOID": "🔴"
        }
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Signal", f"{signal_colors[signal_result['signal']]} {signal_result['signal']}")
        with col2:
            st.metric("Score", f"{signal_result['score']}/100")
        with col3:
            st.metric("Confidence", f"{signal_result['confidence']:.0f}%")
        
        # Display Reasons
        st.subheader("📋 Analysis Details")
        for i, reason in enumerate(signal_result['reasons'], 1):
            st.write(f"{i}. {reason}")
        
        # Action Message
        st.markdown("---")
        st.info(f"**Action:** {signal_result['action']}")
        
        # Data Table
        st.subheader("📊 Full Option Chain Data")
        st.dataframe(df, use_container_width=True)
        
        # Download Data
        csv_buffer = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Analysis CSV",
            data=csv_buffer,
            file_name=f"analysis_{expiry.replace('-', '')}.csv",
            mime="text/csv"
        )
    else:
        st.info("👈 Please upload a CSV file and click 'Analyze' to get started")

if __name__ == "__main__":
    main()
