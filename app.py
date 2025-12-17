import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Import plotly with error handling
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError as e:
    st.error(f"Plotly import error: {e}")
    PLOTLY_AVAILABLE = False

# Page Configuration
st.set_page_config(
    page_title="Option Chain Analyzer - Buyer's Edge",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Buyer-Focused UI
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
    
    .glow-effect {
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { box-shadow: 0 0 5px #fff, 0 0 10px #fff, 0 0 15px #667eea; }
        to { box-shadow: 0 0 10px #fff, 0 0 20px #667eea, 0 0 30px #667eea; }
    }
</style>
""", unsafe_allow_html=True)

# NSE CSV Auto-Converter Function
def auto_convert_nse_csv(raw_df):
    """
    Automatically convert NSE CSV to required format
    Handles real NSE format with CALLS | Strike | PUTS structure
    """
    try:
        st.info("🔄 Detecting NSE CSV structure...")
        
        # NSE format: CALLS section | Strike Price (middle) | PUTS section
        # Find Strike Price column (usually in middle with numeric values)
        
        strike_col_idx = None
        
        # Method 1: Look for column with "strike" in name
        for i, col in enumerate(raw_df.columns):
            col_str = str(col).strip().lower()
            if 'strike' in col_str:
                strike_col_idx = i
                break
        
        # Method 2: Find middle column with numeric strike-like values
        if strike_col_idx is None:
            for i, col in enumerate(raw_df.columns):
                try:
                    sample = pd.to_numeric(raw_df[col].dropna().head(), errors='coerce')
                    if sample.notna().any() and 10000 < sample.max() < 100000:
                        strike_col_idx = i
                        break
                except:
                    continue
        
        # Method 3: Use middle column
        if strike_col_idx is None:
            strike_col_idx = len(raw_df.columns) // 2
        
        st.write(f"✓ Strike Price at column {strike_col_idx + 1}")
        
        # Find OI columns
        # CALLS: before strike, PUTS: after strike
        call_oi_idx = None
        call_chg_idx = None
        put_oi_idx = None
        put_chg_idx = None
        
        # Search CALLS section (before strike)
        for i in range(strike_col_idx - 1, -1, -1):
            col_lower = str(raw_df.columns[i]).lower()
            if 'oi' in col_lower:
                if 'change' not in col_lower and 'chng' not in col_lower and call_oi_idx is None:
                    call_oi_idx = i
                elif ('change' in col_lower or 'chng' in col_lower) and call_chg_idx is None:
                    call_chg_idx = i
        
        # Search PUTS section (after strike)
        for i in range(strike_col_idx + 1, len(raw_df.columns)):
            col_lower = str(raw_df.columns[i]).lower()
            if 'oi' in col_lower:
                if 'change' not in col_lower and 'chng' not in col_lower and put_oi_idx is None:
                    put_oi_idx = i
                elif ('change' in col_lower or 'chng' in col_lower) and put_chg_idx is None:
                    put_chg_idx = i
        
        # If still not found, use position-based guessing
        if call_oi_idx is None:
            call_oi_idx = max(0, strike_col_idx - 2)
        if call_chg_idx is None:
            call_chg_idx = max(0, strike_col_idx - 3)
        if put_oi_idx is None:
            put_oi_idx = min(len(raw_df.columns) - 1, strike_col_idx + 2)
        if put_chg_idx is None:
            put_chg_idx = min(len(raw_df.columns) - 1, strike_col_idx + 3)
        
        st.write(f"✓ Call OI at column {call_oi_idx + 1}")
        st.write(f"✓ Put OI at column {put_oi_idx + 1}")
        
        # Create clean DataFrame
        clean_df = pd.DataFrame()
        clean_df['Strike Price'] = pd.to_numeric(raw_df.iloc[:, strike_col_idx], errors='coerce')
        clean_df['Call OI'] = pd.to_numeric(raw_df.iloc[:, call_oi_idx], errors='coerce').fillna(0)
        clean_df['Call OI Change'] = pd.to_numeric(raw_df.iloc[:, call_chg_idx], errors='coerce').fillna(0)
        clean_df['Put OI'] = pd.to_numeric(raw_df.iloc[:, put_oi_idx], errors='coerce').fillna(0)
        clean_df['Put OI Change'] = pd.to_numeric(raw_df.iloc[:, put_chg_idx], errors='coerce').fillna(0)
        
        # Clean
        clean_df = clean_df.dropna(subset=['Strike Price'])
        clean_df = clean_df[(clean_df['Call OI'] > 0) | (clean_df['Put OI'] > 0)]
        
        st.success(f"✅ Converted successfully! Found {len(clean_df)} strikes")
        
        # Show summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Strikes", len(clean_df))
        with col2:
            st.metric("Call OI", f"{clean_df['Call OI'].sum():,.0f}")
        with col3:
            st.metric("Put OI", f"{clean_df['Put OI'].sum():,.0f}")
        
        return clean_df
        
    except Exception as e:
        st.error(f"❌ Auto-conversion failed: {e}")
        st.info("💡 Try using the standalone converter script or check CSV format")
        return None

# Signal Engine Class
class OptionChainAnalyzer:
    def __init__(self, df):
        self.df = df
        self.spot_price = None
        self.max_pain = None
        self.pcr = None
        
    def calculate_max_pain(self):
        """Calculate Max Pain Strike"""
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
        except Exception as e:
            st.warning(f"Max Pain calculation error: {e}")
            return None
    
    def calculate_pcr(self):
        """Calculate Put-Call Ratio"""
        try:
            total_put_oi = self.df['Put OI'].sum()
            total_call_oi = self.df['Call OI'].sum()
            self.pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 1.0
            return self.pcr
        except Exception as e:
            st.warning(f"PCR calculation error: {e}")
            return 1.0
    
    def detect_market_type(self):
        """Classify Market: TREND/EXPANSION vs RANGE/CHOP"""
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
    
    def calculate_break_probability(self, strike):
        """Calculate probability of breaking through a strike"""
        try:
            strike_data = self.df[self.df['Strike Price'] == strike]
            if strike_data.empty:
                return 0
            
            if self.spot_price > strike:
                oi_change = strike_data['Call OI Change'].values[0]
                total_oi = strike_data['Call OI'].values[0]
            else:
                oi_change = strike_data['Put OI Change'].values[0]
                total_oi = strike_data['Put OI'].values[0]
            
            unwinding_pct = max(0, -oi_change / total_oi * 100) if total_oi > 0 else 0
            probability = min(100, (unwinding_pct * 0.4 + 30))
            return round(probability, 1)
        except:
            return 0
    
    def generate_signal(self):
        """Generate BUY/WAIT/AVOID Signal"""
        score = 0
        reasons = []
        direction = None
        
        current_hour = datetime.now().hour
        
        # Market Type Filter
        market_type = self.detect_market_type()
        
        if market_type == "RANGE":
            return {
                'signal': 'AVOID',
                'score': 0,
                'confidence': 0,
                'market_type': market_type,
                'direction': None,
                'best_strike': None,
                'reasons': ['Market in RANGE/CHOP zone', 'Theta decay will kill premium', 'Wait for clear directional move'],
                'action': 'Stay out of the market'
            }
        
        # Time Window Check (10 points)
        if 11 <= current_hour < 15:
            score += 10
            reasons.append('✓ Trading in optimal time window')
        else:
            reasons.append('✗ Outside optimal trading hours')
        
        # VWAP Alignment (20 points)
        if self.spot_price and self.max_pain:
            if self.spot_price > self.max_pain:
                score += 20
                reasons.append('✓ Price above key level (Bullish)')
                direction = "CALLS"
            else:
                score += 15
                reasons.append('✓ Price below key level (Bearish)')
                direction = "PUTS"
        
        # OI Analysis (45 points)
        total_call_oi_change = self.df['Call OI Change'].sum()
        total_put_oi_change = self.df['Put OI Change'].sum()
        
        if direction == "CALLS" and total_put_oi_change < 0:
            score += 25
            reasons.append('✓ PUT OI unwinding (Support weakening)')
        elif direction == "PUTS" and total_call_oi_change < 0:
            score += 25
            reasons.append('✓ CALL OI unwinding (Resistance weakening)')
        
        if direction == "CALLS" and total_call_oi_change > 0:
            score += 20
            reasons.append('✓ Fresh CALL OI addition (Bullish conviction)')
        elif direction == "PUTS" and total_put_oi_change > 0:
            score += 20
            reasons.append('✓ Fresh PUT OI addition (Bearish conviction)')
        
        # Max Pain Distance (15 points)
        if self.spot_price and self.max_pain:
            distance_pct = abs(self.spot_price - self.max_pain) / self.spot_price * 100
            if distance_pct > 0.35:
                score += 15
                reasons.append(f'✓ Good distance from Max Pain ({distance_pct:.2f}%)')
        
        score += 5  # Volume placeholder
        
        # Determine Signal
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
        """Find optimal strike"""
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
        """Generate actionable message"""
        if signal == "BUY" and best_strike:
            return f"BUY {best_strike['strike']} {best_strike['type']} with strict stop-loss"
        elif signal == "WAIT":
            return "Wait for clear price confirmation and volume expansion"
        else:
            return "No trade setup - Protect capital and wait"

# Main Application
def main():
    st.markdown('<h1 class="main-header">📊 OPTION CHAIN ANALYZER - BUYER\'S EDGE</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Configuration")
        uploaded_file = st.file_uploader("Upload NSE Option Chain CSV", type=['csv'])
        
        st.markdown("---")
        st.markdown("### 📌 Quick Guide")
        st.info("**🟢 BUY:** High conviction\n**🟡 WAIT:** Setup forming\n**🔴 AVOID:** No trade zone")
    
    if uploaded_file is None:
        st.info("👆 Please upload an NSE Option Chain CSV file")
        
        st.markdown("---")
        st.subheader("📋 Expected CSV Format")
        sample_df = pd.DataFrame({
            'Strike Price': [59000, 59100, 59200, 59300, 59400],
            'Call OI': [150000, 200000, 180000, 120000, 80000],
            'Call OI Change': [-5000, 15000, 20000, -3000, 2000],
            'Put OI': [80000, 120000, 180000, 200000, 150000],
            'Put OI Change': [2000, -3000, 20000, 15000, -5000],
        })
        st.dataframe(sample_df, use_container_width=True)
        return
    
    try:
        # Try to read raw CSV
        raw_df = pd.read_csv(uploaded_file)
        
        # Check if already in correct format
        required_columns = ['Strike Price', 'Call OI', 'Put OI']
        columns_lower = [col.strip().lower() for col in raw_df.columns]
        
        needs_conversion = not all(
            any(req.lower() in col for col in columns_lower) 
            for req in required_columns
        )
        
        if needs_conversion:
            # Auto-convert NSE format
            st.info("🔄 Converting NSE CSV format to analysis format...")
            df = auto_convert_nse_csv(raw_df)
            if df is None:
                st.error("❌ Unable to convert CSV. Please check format.")
                st.info("💡 Expected columns: Strike Price, Call OI, Put OI (and optionally Call/Put OI Change)")
                return
        else:
            df = raw_df
            # Standardize column names
            col_mapping = {}
            for col in df.columns:
                col_lower = col.strip().lower()
                if 'strike' in col_lower and 'price' in col_lower:
                    col_mapping[col] = 'Strike Price'
                elif col_lower == 'strike':
                    col_mapping[col] = 'Strike Price'
                elif 'call' in col_lower and 'oi' in col_lower and 'change' not in col_lower:
                    col_mapping[col] = 'Call OI'
                elif 'call' in col_lower and 'change' in col_lower:
                    col_mapping[col] = 'Call OI Change'
                elif 'put' in col_lower and 'oi' in col_lower and 'change' not in col_lower:
                    col_mapping[col] = 'Put OI'
                elif 'put' in col_lower and 'change' in col_lower:
                    col_mapping[col] = 'Put OI Change'
            df = df.rename(columns=col_mapping)
        
        # Ensure OI Change columns exist
        if 'Call OI Change' not in df.columns:
            df['Call OI Change'] = 0
        if 'Put OI Change' not in df.columns:
            df['Put OI Change'] = 0
        
        # Clean data
        df = df.dropna(subset=['Strike Price'])
        for col in ['Strike Price', 'Call OI', 'Put OI', 'Call OI Change', 'Put OI Change']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df = df[(df['Call OI'] > 0) | (df['Put OI'] > 0)]
        df = df.sort_values('Strike Price').reset_index(drop=True)
        
        # Initialize Analyzer
        analyzer = OptionChainAnalyzer(df)
        analyzer.spot_price = df['Strike Price'].median()
        analyzer.max_pain = analyzer.calculate_max_pain()
        analyzer.pcr = analyzer.calculate_pcr()
        
        # Generate Signal
        signal_result = analyzer.generate_signal()
        
        # Display Signal
        if signal_result['signal'] == 'BUY':
            st.markdown(f"""
            <div class="buy-signal glow-effect">
                🟢 BUY {signal_result['direction']}
                <div style="font-size: 1.2rem; margin-top: 10px;">
                    Confidence: {signal_result['confidence']}% | Market: {signal_result['market_type']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif signal_result['signal'] == 'WAIT':
            st.markdown(f"""
            <div class="wait-signal">
                🟡 WAIT FOR CONFIRMATION
                <div style="font-size: 1.2rem; margin-top: 10px;">
                    Score: {signal_result['score']}/100 | Market: {signal_result['market_type']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="avoid-signal">
                🔴 AVOID - NO TRADE ZONE
                <div style="font-size: 1.2rem; margin-top: 10px;">
                    Market: {signal_result['market_type']} | Buyer Risk: EXTREME
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Action
        st.markdown(f"""
        <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #667eea; margin: 20px 0;">
            <h3 style="margin: 0; color: #333;">📍 ACTION REQUIRED</h3>
            <p style="font-size: 1.2rem; margin: 10px 0 0 0;">{signal_result['action']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Signal Score", f"{signal_result['score']}/100")
        with col2:
            st.metric("Market Type", signal_result['market_type'])
        with col3:
            st.metric("Max Pain", f"₹{analyzer.max_pain:.0f}" if analyzer.max_pain else "N/A")
        
        # Reasons
        st.markdown("### 🔍 Signal Breakdown")
        for reason in signal_result['reasons']:
            if '✓' in reason:
                st.success(reason)
            else:
                st.warning(reason)
        
        # Best Strike
        if signal_result['best_strike']:
            st.markdown("---")
            st.markdown("### 🎯 RECOMMENDED STRIKE")
            strike_info = signal_result['best_strike']
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Strike", f"₹{strike_info['strike']} {strike_info['type']}")
            with col2:
                st.metric("OI Change", f"{strike_info['oi_change']:,}")
            with col3:
                st.metric("Total OI", f"{strike_info['oi']:,}")
        
        # Charts (only if Plotly available)
        if PLOTLY_AVAILABLE:
            st.markdown("---")
            st.markdown("### 📊 VISUAL ANALYSIS")
            
            # Simple bar chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df['Strike Price'],
                y=df['Call OI'],
                name='Call OI',
                marker_color='red'
            ))
            fig.add_trace(go.Bar(
                x=df['Strike Price'],
                y=df['Put OI'],
                name='Put OI',
                marker_color='green'
            ))
            fig.update_layout(title='OI Distribution', height=500, barmode='group')
            st.plotly_chart(fig, use_container_width=True)
        
        # Data Table
        st.markdown("---")
        st.markdown("### 📋 DETAILED DATA")
        st.dataframe(df, use_container_width=True, height=400)
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("Ensure CSV has: Strike Price, Call OI, Put OI")

if __name__ == "__main__":
    main()
