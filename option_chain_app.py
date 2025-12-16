import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import numpy as np

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
    
    .trade-zone-card {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
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
    
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    .strike-row {
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    
    .strike-row:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
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

# Signal Engine Functions
class OptionChainAnalyzer:
    def __init__(self, df):
        self.df = df
        self.spot_price = None
        self.max_pain = None
        self.pcr = None
        self.signals = {}
        
    def calculate_max_pain(self):
        """Calculate Max Pain Strike"""
        try:
            total_pain = []
            strikes = self.df['Strike Price'].unique()
            
            for strike in strikes:
                call_pain = self.df[self.df['Strike Price'] < strike]['Call OI'].sum() * (strike - self.df[self.df['Strike Price'] < strike]['Strike Price']).sum()
                put_pain = self.df[self.df['Strike Price'] > strike]['Put OI'].sum() * (self.df[self.df['Strike Price'] > strike]['Strike Price'] - strike).sum()
                total_pain.append(call_pain + put_pain)
            
            self.max_pain = strikes[np.argmin(total_pain)]
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
        """Classify Market: TREND/EXPANSION vs RANGE/CHOP"""
        if self.spot_price is None or self.max_pain is None:
            return "UNKNOWN"
        
        # Calculate distance from max pain
        distance_pct = abs(self.spot_price - self.max_pain) / self.spot_price * 100
        
        # Check OI Change dominance
        total_call_oi_change = abs(self.df['Call OI Change'].sum())
        total_put_oi_change = abs(self.df['Put OI Change'].sum())
        oi_dominance = max(total_call_oi_change, total_put_oi_change) / (total_call_oi_change + total_put_oi_change)
        
        # Classification
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
            
            # Factors influencing break probability
            factors = {
                'oi_unwinding': 0,
                'oi_addition_opposite': 0,
                'price_momentum': 0,
                'volume': 0
            }
            
            # Check OI unwinding at strike
            if self.spot_price > strike:  # Checking call strike
                factors['oi_unwinding'] = max(0, -strike_data['Call OI Change'].values[0] / strike_data['Call OI'].values[0] * 100) if strike_data['Call OI'].values[0] > 0 else 0
            else:  # Checking put strike
                factors['oi_unwinding'] = max(0, -strike_data['Put OI Change'].values[0] / strike_data['Put OI'].values[0] * 100) if strike_data['Put OI'].values[0] > 0 else 0
            
            # Normalize and calculate probability
            probability = min(100, (factors['oi_unwinding'] * 0.4 + 30))
            return round(probability, 1)
        except:
            return 0
    
    def generate_signal(self):
        """Generate BUY/WAIT/AVOID Signal"""
        score = 0
        reasons = []
        signal_type = "AVOID"
        
        # Get current time (simulated for demo)
        current_hour = datetime.now().hour
        current_minute = datetime.now().minute
        
        # Market Type Filter (FIRST GATE)
        market_type = self.detect_market_type()
        
        if market_type == "RANGE":
            return {
                'signal': 'AVOID',
                'score': 0,
                'confidence': 0,
                'market_type': market_type,
                'reasons': ['Market in RANGE/CHOP zone', 'Theta decay will kill option premium', 'Wait for clear directional move'],
                'action': 'Stay out of the market'
            }
        
        # Time Window Check (10 points)
        if 11 <= current_hour < 15:
            score += 10
            reasons.append('✓ Trading in optimal time window')
        else:
            reasons.append('✗ Outside optimal trading hours')
        
        # VWAP Alignment (20 points)
        # For demo, assuming spot is VWAP proxy
        if self.spot_price and self.max_pain:
            if self.spot_price > self.max_pain:
                score += 20
                reasons.append('✓ Price above key level (Bullish structure)')
                direction = "CALLS"
            else:
                score += 15
                reasons.append('✓ Price below key level (Bearish structure)')
                direction = "PUTS"
        
        # OI Analysis (45 points total)
        total_call_oi_change = self.df['Call OI Change'].sum()
        total_put_oi_change = self.df['Put OI Change'].sum()
        
        # OI Unwinding (25 points)
        if direction == "CALLS" and total_put_oi_change < 0:
            score += 25
            reasons.append('✓ PUT OI unwinding (Support weakening)')
        elif direction == "PUTS" and total_call_oi_change < 0:
            score += 25
            reasons.append('✓ CALL OI unwinding (Resistance weakening)')
        
        # OI Addition opposite side (20 points)
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
        
        # Volume Expansion (10 points) - Simulated
        score += 5
        
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
        
        # Find best strike
        best_strike = self.find_best_strike(direction if score >= 50 else None)
        
        return {
            'signal': signal_type,
            'score': score,
            'confidence': confidence,
            'market_type': market_type,
            'direction': direction if score >= 50 else None,
            'best_strike': best_strike,
            'reasons': reasons,
            'action': self.get_action_message(signal_type, direction if score >= 50 else None, best_strike)
        }
    
    def find_best_strike(self, direction):
        """Find optimal strike for buying"""
        if not direction or self.spot_price is None:
            return None
        
        # Find ATM strike
        self.df['Distance'] = abs(self.df['Strike Price'] - self.spot_price)
        atm_strike = self.df.loc[self.df['Distance'].idxmin(), 'Strike Price']
        
        # Get ATM ±1 strikes
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
    # Header
    st.markdown('<h1 class="main-header">📊 OPTION CHAIN ANALYZER - BUYER\'S EDGE</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/stock-share.png", width=80)
        st.title("⚙️ Configuration")
        
        uploaded_file = st.file_uploader("Upload NSE Option Chain CSV", type=['csv'])
        
        st.markdown("---")
        st.markdown("### 📌 Quick Guide")
        st.info("""
        **🟢 BUY:** High conviction trade setup  
        **🟡 WAIT:** Setup forming, need confirmation  
        **🔴 AVOID:** No trade zone - stay out
        """)
        
        st.markdown("---")
        st.markdown("### 🎯 Buyer's Checklist")
        st.markdown("""
        - ✅ Only trade EXPANSION markets
        - ✅ Focus on OI Change > Static OI
        - ✅ Avoid Max Pain zones
        - ✅ Trade between 11:30 - 14:45
        - ✅ Use ATM ±1 strikes only
        """)
    
    # Main Content
    if uploaded_file is None:
        st.info("👆 Please upload an NSE Option Chain CSV file to begin analysis")
        
        # Demo Instructions
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
        st.caption("Your CSV should contain: Strike Price, Call OI, Call OI Change, Put OI, Put OI Change")
        
        return
    
    # Load Data
    try:
        df = pd.read_csv(uploaded_file)
        
        # Data validation
        required_columns = ['Strike Price', 'Call OI', 'Put OI']
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            return
        
        # Add OI Change columns if not present
        if 'Call OI Change' not in df.columns:
            df['Call OI Change'] = 0
        if 'Put OI Change' not in df.columns:
            df['Put OI Change'] = 0
        
        # Initialize Analyzer
        analyzer = OptionChainAnalyzer(df)
        
        # Calculate metrics
        analyzer.spot_price = df['Strike Price'].median()  # Approximation
        analyzer.max_pain = analyzer.calculate_max_pain()
        analyzer.pcr = analyzer.calculate_pcr()
        
        # Generate Signal
        signal_result = analyzer.generate_signal()
        
        # Display Signal Card
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
                    Setup Score: {signal_result['score']}/100 | Market: {signal_result['market_type']}
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
        
        # Trade Action
        st.markdown(f"""
        <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #667eea; margin: 20px 0;">
            <h3 style="margin: 0; color: #333;">📍 ACTION REQUIRED</h3>
            <p style="font-size: 1.2rem; margin: 10px 0 0 0; color: #555;">{signal_result['action']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed Breakdown
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Signal Score", f"{signal_result['score']}/100", 
                     delta="Tradeable" if signal_result['score'] >= 75 else "Not Ready")
        
        with col2:
            st.metric("Market Type", signal_result['market_type'],
                     delta="Favorable" if signal_result['market_type'] == "EXPANSION" else "Risky")
        
        with col3:
            st.metric("Max Pain", f"₹{analyzer.max_pain:.0f}" if analyzer.max_pain else "N/A",
                     delta=f"{abs(analyzer.spot_price - analyzer.max_pain):.0f} pts away" if analyzer.max_pain else None)
        
        # Reasons
        st.markdown("### 🔍 Signal Breakdown")
        for reason in signal_result['reasons']:
            if '✓' in reason:
                st.success(reason)
            else:
                st.warning(reason)
        
        # Best Strike Recommendation
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
        
        # Charts
        st.markdown("---")
        st.markdown("### 📊 VISUAL ANALYSIS")
        
        tab1, tab2, tab3 = st.tabs(["🔥 Strike Heatmap", "📈 OI Distribution", "🎯 PCR Analysis"])
        
        with tab1:
            # Strike Heatmap
            fig = go.Figure()
            
            # Normalize OI for color intensity
            max_call_oi = df['Call OI'].max()
            max_put_oi = df['Put OI'].max()
            
            fig.add_trace(go.Bar(
                y=df['Strike Price'],
                x=-df['Put OI'],
                name='Put OI',
                orientation='h',
                marker=dict(
                    color=df['Put OI Change'],
                    colorscale='Greens',
                    showscale=True,
                    colorbar=dict(title="PUT OI Change", x=-0.15)
                ),
                hovertemplate='Strike: %{y}<br>Put OI: %{x:,}<extra></extra>'
            ))
            
            fig.add_trace(go.Bar(
                y=df['Strike Price'],
                x=df['Call OI'],
                name='Call OI',
                orientation='h',
                marker=dict(
                    color=df['Call OI Change'],
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="CALL OI Change", x=1.15)
                ),
                hovertemplate='Strike: %{y}<br>Call OI: %{x:,}<extra></extra>'
            ))
            
            fig.update_layout(
                title='Strike-wise OI Distribution with Change Intensity',
                barmode='overlay',
                height=600,
                xaxis_title='Open Interest',
                yaxis_title='Strike Price',
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # OI Change Analysis
            fig2 = go.Figure()
            
            fig2.add_trace(go.Scatter(
                x=df['Strike Price'],
                y=df['Call OI Change'],
                name='Call OI Change',
                mode='lines+markers',
                line=dict(color='red', width=3),
                fill='tozeroy'
            ))
            
            fig2.add_trace(go.Scatter(
                x=df['Strike Price'],
                y=df['Put OI Change'],
                name='Put OI Change',
                mode='lines+markers',
                line=dict(color='green', width=3),
                fill='tozeroy'
            ))
            
            fig2.update_layout(
                title='OI Change Flow (Most Critical for Buyers)',
                height=500,
                xaxis_title='Strike Price',
                yaxis_title='OI Change',
                template='plotly_white',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        
        with tab3:
            # PCR Gauge
            fig3 = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=analyzer.pcr,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Put-Call Ratio", 'font': {'size': 24}},
                delta={'reference': 1.0, 'increasing': {'color': "green"}},
                gauge={
                    'axis': {'range': [None, 2], 'tickwidth': 1},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'steps': [
                        {'range': [0, 0.7], 'color': 'red'},
                        {'range': [0.7, 1.3], 'color': 'yellow'},
                        {'range': [1.3, 2], 'color': 'green'}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 1.0
                    }
                }
            ))
            
            fig3.update_layout(height=400)
            st.plotly_chart(fig3, use_container_width=True)
            
            st.caption("PCR between 0.9-1.1 indicates range-bound market (AVOID for option buyers)")
        
        # Raw Data Table
        st.markdown("---")
        st.markdown("### 📋 DETAILED STRIKE DATA")
        
        # Add color coding to dataframe
        styled_df = df.style.background_gradient(subset=['Call OI'], cmap='Reds')\
                            .background_gradient(subset=['Put OI'], cmap='Greens')\
                            .format({
                                'Call OI': '{:,.0f}',
                                'Put OI': '{:,.0f}',
                                'Call OI Change': '{:+,.0f}',
                                'Put OI Change': '{:+,.0f}'
                            })
        
        st.dataframe(styled_df, use_container_width=True, height=400)
        
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please ensure your CSV has the correct format with columns: Strike Price, Call OI, Put OI")

if __name__ == "__main__":
    main()
