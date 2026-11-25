import asyncio
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.portfolio_opt import SentimentPortfolioOptimizer
from src.risk_engine import QuantitativeRiskEngine
from src.sentiment_ai import AsyncSentimentEngine

st.set_page_config(
    page_title="AI-Driven Market Risk & Sentiment Engine",
    page_icon="📊",
    layout="wide",
)

st.title("📊 AI-Driven Market Risk & Sentiment Engine")
st.markdown(
    "Integrating **Historical, Parametric & Monte Carlo VaR/CVaR** with **LLM Financial Sentiment (Llama / Groq + Pydantic)** for dynamic portfolio risk management."
)


@st.cache_resource
def load_risk_engine() -> QuantitativeRiskEngine:
    return QuantitativeRiskEngine()


risk_engine = load_risk_engine()
optimizer = SentimentPortfolioOptimizer(risk_engine)

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Portfolio & Risk Parameters")

confidence = st.sidebar.select_slider(
    "Confidence Level (VaR / CVaR)",
    options=[0.90, 0.95, 0.99],
    value=0.95,
    format_func=lambda x: f"{int(x * 100)}%",
)

sentiment_tilt = st.sidebar.slider(
    "AI Sentiment Tilt Sensitivity (λ)",
    min_value=0.0,
    max_value=0.40,
    value=0.15,
    step=0.05,
    help="Controls how strongly LLM news sentiment tilts expected returns.",
)

st.sidebar.subheader("Initial Asset Allocation")
w_spy = st.sidebar.slider("SPY (US Equities)", 0.0, 0.60, 0.30, 0.05)
w_mib = st.sidebar.slider("FTSEMIB.MI (Italy Equities)", 0.0, 0.60, 0.25, 0.05)
w_oil = st.sidebar.slider("CL=F (Crude Oil)", 0.0, 0.60, 0.20, 0.05)
w_gas = st.sidebar.slider("NG=F (Natural Gas)", 0.0, 0.60, 0.15, 0.05)
w_gld = st.sidebar.slider("GC=F (Gold)", 0.0, 0.60, 0.10, 0.05)

raw_weights = {
    "SPY": w_spy,
    "FTSEMIB.MI": w_mib,
    "CL=F": w_oil,
    "NG=F": w_gas,
    "GC=F": w_gld,
}
total_w = sum(raw_weights.values())
if total_w == 0:
    st.error("Please assign positive weight to at least one asset.")
    st.stop()

base_weights = {k: round(v / total_w, 4) for k, v in raw_weights.items()}
if not np.isclose(total_w, 1.0):
    st.sidebar.info(f"Weights auto-normalized from {total_w:.0%} to 100%.")

# --- Interactive News Feed Input ---
st.subheader("📰 Live Market & Energy News Feed (Editable)")
default_news = [
    {"ticker": "SPY", "headline": "US tech earnings beat expectations amid resilient consumer demand growth"},
    {"ticker": "FTSEMIB.MI", "headline": "European industrial output faces recession risk as borrowing costs stay elevated"},
    {"ticker": "CL=F", "headline": "Global crude oil prices drop sharply following unexpected surge in supply"},
    {"ticker": "NG=F", "headline": "European natural gas faces severe supply disruption and geopolitical sanctions"},
    {"ticker": "GC=F", "headline": "Gold prices rally to record high as investors hedge geopolitical risk"},
]

news_df = st.data_editor(
    pd.DataFrame(default_news),
    use_container_width=True,
    num_rows="fixed",
    column_config={
        "ticker": st.column_config.TextColumn("Asset Ticker", disabled=True),
        "headline": st.column_config.TextColumn("Market Headline (Edit to test AI reaction)", width="large"),
    },
)

if st.button("🚀 Run AI Risk & Portfolio Optimization", type="primary"):
    with st.spinner("Running async LLM sentiment inference & Monte Carlo simulations..."):
        sent_engine = AsyncSentimentEngine()
        news_list = news_df.to_dict(orient="records")
        signals = asyncio.run(sent_engine.analyze_batch(news_list))

        port_ret = risk_engine.compute_portfolio_returns(base_weights)
        hist_risk = risk_engine.historical_var_cvar(port_ret, confidence=confidence)
        base_mc = risk_engine.monte_carlo_var_cvar(base_weights, confidence=confidence)
        kupiec = risk_engine.kupiec_backtest(port_ret, hist_risk["VaR"], confidence=confidence)

        opt_weights, comp_df = optimizer.optimize_weights(
            base_weights, signals, sentiment_weight=sentiment_tilt
        )
        opt_mc = risk_engine.monte_carlo_var_cvar(opt_weights, confidence=confidence)

        # --- KPI Row ---
        st.divider()
        st.subheader(f"📈 Quantitative Risk Summary ({int(confidence * 100)}% 1-Day Horizon)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Baseline Monte Carlo VaR",
            f"{base_mc['VaR']:.2%}",
            delta=f"{(opt_mc['VaR'] - base_mc['VaR']):.2%} after AI",
            delta_color="inverse",
        )
        c2.metric(
            "Baseline Monte Carlo CVaR",
            f"{base_mc['CVaR']:.2%}",
            delta=f"{(opt_mc['CVaR'] - base_mc['CVaR']):.2%} after AI",
            delta_color="inverse",
        )
        c3.metric(
            "Historical CVaR (Fat-Tail)",
            f"{hist_risk['CVaR']:.2%}",
            help="Captures non-normal energy & commodity tail shocks.",
        )
        c4.metric(
            "Kupiec Backtest p-value",
            f"{kupiec['kupiec_p_value']:.3f}",
            delta="Model Accepted" if kupiec["model_accepted"] else "Model Rejected",
            delta_color="normal" if kupiec["model_accepted"] else "inverse",
        )

        # --- Charts Row ---
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### Portfolio Return Distribution & Tail Risk")
            fig_hist = px.histogram(
                port_ret,
                nbins=50,
                labels={"value": "Daily Log Return", "count": "Frequency"},
                color_discrete_sequence=["#3b82f6"],
            )
            fig_hist.add_vline(
                x=-hist_risk["VaR"],
                line_dash="dash",
                line_color="orange",
                annotation_text=f"VaR ({hist_risk['VaR']:.2%})",
                annotation_position="top right",
            )
            fig_hist.add_vline(
                x=-hist_risk["CVaR"],
                line_dash="solid",
                line_color="red",
                annotation_text=f"CVaR ({hist_risk['CVaR']:.2%})",
                annotation_position="top left",
            )
            fig_hist.update_layout(showlegend=False, height=380)
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_right:
            st.markdown("#### Asset Allocation: Base vs. AI-Optimized")
            chart_df = pd.DataFrame({
                "Asset": list(base_weights.keys()) * 2,
                "Weight": [base_weights[t] * 100 for t in base_weights] + [opt_weights[t] * 100 for t in opt_weights],
                "Strategy": ["Base Portfolio"] * len(base_weights) + ["AI-Adjusted Portfolio"] * len(opt_weights),
            })
            fig_bar = px.bar(
                chart_df,
                x="Asset",
                y="Weight",
                color="Strategy",
                barmode="group",
                text_auto=".1f",
                labels={"Weight": "Allocation (%)"},
                color_discrete_map={"Base Portfolio": "#94a3b8", "AI-Adjusted Portfolio": "#10b981"},
            )
            fig_bar.update_layout(height=380)
            st.plotly_chart(fig_bar, use_container_width=True)

        # --- AI Signals & Rebalancing Table ---
        st.subheader("🤖 Structured LLM Sentiment Signals & Rationale")
        sig_rows = []
        for t, sig in signals.items():
            sig_rows.append({
                "Asset": t,
                "Base Weight": f"{base_weights[t]:.1%}",
                "AI-Optimized Weight": f"{opt_weights[t]:.1%}",
                "Sentiment Score": f"{sig.sentiment_score:+.2f}",
                "Regime": sig.regime,
                "Confidence": f"{sig.confidence:.0%}",
                "AI Financial Rationale": sig.rationale,
            })
        st.dataframe(pd.DataFrame(sig_rows), use_container_width=True, hide_index=True)