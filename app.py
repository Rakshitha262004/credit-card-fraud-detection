"""
app.py — Credit Card Fraud Detection Dashboard
Streamlit Live Simulation Dashboard (Dark Banking Theme)

Run with:
    streamlit run app.py

Author  : Your Name
Project : Credit Card Fraud Detection System
"""

import time
import random
import joblib
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield · SecureBank",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS — dark banking terminal theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Share Tech Mono', monospace !important;
    background-color: #060d18 !important;
    color: #8ab4d4 !important;
}

/* ── Main area background ── */
.stApp {
    background: #060d18;
    background-image:
        linear-gradient(rgba(0,180,255,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,180,255,0.025) 1px, transparent 1px);
    background-size: 40px 40px;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #07101e !important;
    border-right: 1px solid #0d2035 !important;
}
section[data-testid="stSidebar"] * { color: #8ab4d4 !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #080f1c;
    border: 1px solid #0d2035;
    border-top: 2px solid #00ccff;
    border-radius: 4px;
    padding: 12px !important;
}
[data-testid="metric-container"] label {
    color: #3a6080 !important;
    font-size: 10px !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #00ccff !important;
    font-size: 22px !important;
    font-family: 'Orbitron', monospace !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 11px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #003355, #005588) !important;
    color: #00ccff !important;
    border: 1px solid #00aaff44 !important;
    border-radius: 4px !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 0.1em !important;
    padding: 8px 20px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    border-color: #00ccff !important;
    box-shadow: 0 0 12px #00ccff44 !important;
}

/* ── Selectbox / Slider ── */
.stSelectbox > div, .stSlider > div { color: #8ab4d4 !important; }
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #00ccff !important;
}

/* ── Dataframe ── */
.stDataFrame { border: 1px solid #0d2035 !important; }
.stDataFrame thead th {
    background: #07101e !important;
    color: #3a6080 !important;
    font-size: 10px !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
}
.stDataFrame tbody tr:hover { background: #0d1f35 !important; }

/* ── Headers ── */
h1, h2, h3 {
    font-family: 'Orbitron', monospace !important;
    color: #00ccff !important;
    letter-spacing: 0.08em !important;
}

/* ── Dividers ── */
hr { border-color: #0d2035 !important; }

/* ── Info / Warning / Error boxes ── */
.stAlert { border-radius: 4px !important; font-family: 'Share Tech Mono', monospace !important; }

/* ── Expander ── */
details { border: 1px solid #0d2035 !important; border-radius: 4px !important; }
summary { color: #3a6080 !important; font-size: 11px !important; letter-spacing: 0.1em !important; }

/* ── Tab bar ── */
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #3a6080 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.1em !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #00ccff !important;
    border-bottom: 2px solid #00ccff !important;
}

/* ── Plotly chart container ── */
.js-plotly-plot { border: 1px solid #0d2035; border-radius: 4px; }

/* ── Status badges ── */
.badge-fraud {
    background: #ff3a3a22; border: 1px solid #ff3a3a55;
    color: #ff3a3a; padding: 2px 8px; border-radius: 3px;
    font-size: 11px; font-weight: bold;
}
.badge-legit {
    background: #00ff8822; border: 1px solid #00ff8855;
    color: #00ff88; padding: 2px 8px; border-radius: 3px;
    font-size: 11px;
}
.section-header {
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px; letter-spacing: 0.2em;
    color: #3a6080; text-transform: uppercase;
    border-bottom: 1px solid #0d2035;
    padding-bottom: 4px; margin-bottom: 12px;
}
.alert-card {
    background: #100808;
    border: 1px solid #ff3a3a33;
    border-left: 3px solid #ff3a3a;
    border-radius: 4px; padding: 10px 14px; margin-bottom: 8px;
}
.kpi-label { font-size: 9px; color: #3a6080; letter-spacing: 0.15em; }
.kpi-value { font-size: 20px; color: #00ccff; font-family: 'Orbitron', monospace; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
MERCHANTS = [
    "Amazon", "Flipkart", "Zomato", "Swiggy", "Netflix",
    "Uber", "BigBasket", "MakeMyTrip", "Myntra", "BookMyShow",
    "Shell Petrol", "IRCTC", "PhonePe", "Paytm", "Meesho",
]
FRAUD_MERCHANTS = [
    "Unknown Store", "Intl Wire Transfer", "Crypto Exchange",
    "Offshore ATM", "VPN Purchase", "Anonymous Pay", "Dark Market",
]
LOCATIONS   = ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Hyderabad", "Pune", "Kolkata"]
FRAUD_LOCS  = ["Lagos, NG", "Minsk, BY", "Bucharest, RO", "Unknown", "TOR Node"]

PLOT_DARK = dict(
    paper_bgcolor="#080f1c",
    plot_bgcolor="#060d18",
    font=dict(color="#8ab4d4", family="Share Tech Mono, monospace", size=11),
    xaxis=dict(gridcolor="#0d2035", zerolinecolor="#0d2035"),
    yaxis=dict(gridcolor="#0d2035", zerolinecolor="#0d2035"),
    margin=dict(l=40, r=20, t=40, b=40),
)

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
defaults = dict(
    transactions=[],
    total=0, fraud=0, legit=0,
    total_amount=0.0, fraud_amount=0.0,
    fraud_history=[],   # list of 0/1 per txn
    txn_counter=1000,
    running=False,
    alerts=[],          # last N fraud txns
)
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# TRANSACTION GENERATOR
# ─────────────────────────────────────────────
def generate_transaction(fraud_rate: float = 0.18) -> dict:
    is_fraud = random.random() < fraud_rate
    st.session_state.txn_counter += 1

    if is_fraud:
        amount   = random.choice([
            round(random.uniform(0.1, 1.9), 2),           # card testing
            round(random.uniform(800, 2500), 2),           # large fraud
        ])
        hour     = random.choice([1, 2, 3, 4, 23])
        merchant = random.choice(FRAUD_MERCHANTS)
        location = random.choice(FRAUD_LOCS)
        prob     = round(random.uniform(0.72, 0.99), 3)
    else:
        amount   = round(random.lognormvariate(3.5, 1.0), 2)
        amount   = min(amount, 750)
        hour     = random.randint(8, 21)
        merchant = random.choice(MERCHANTS)
        location = random.choice(LOCATIONS)
        prob     = round(random.uniform(0.01, 0.18), 3)

    # Derive fraud triggers
    triggers = []
    if is_fraud:
        if hour < 6:            triggers.append("Unusual Hour")
        if amount < 2:          triggers.append("Micro Amount")
        if amount > 800:        triggers.append("High Value")
        if location in FRAUD_LOCS: triggers.append("Suspicious Location")
        if prob > 0.9:          triggers.append("Extreme Risk Score")
        if not triggers:        triggers.append("Anomalous Pattern")

    return dict(
        id        = f"TXN{st.session_state.txn_counter}",
        merchant  = merchant,
        amount    = amount,
        location  = location,
        hour      = hour,
        prob      = prob,
        predicted = 1 if is_fraud else 0,
        actual    = 1 if is_fraud else 0,
        card      = f"•••• {random.randint(1000, 9999)}",
        timestamp = datetime.now().strftime("%H:%M:%S"),
        triggers  = ", ".join(triggers) if triggers else "—",
        status    = "🚨 BLOCKED" if is_fraud else "✅ APPROVED",
    )

# ─────────────────────────────────────────────
# UPDATE STATE with a batch of transactions
# ─────────────────────────────────────────────
def process_batch(n: int, fraud_rate: float):
    for _ in range(n):
        txn = generate_transaction(fraud_rate)
        st.session_state.transactions.insert(0, txn)
        st.session_state.total       += 1
        st.session_state.total_amount = round(st.session_state.total_amount + txn["amount"], 2)
        if txn["predicted"] == 1:
            st.session_state.fraud        += 1
            st.session_state.fraud_amount  = round(st.session_state.fraud_amount + txn["amount"], 2)
            st.session_state.alerts.insert(0, txn)
            st.session_state.alerts = st.session_state.alerts[:8]
        else:
            st.session_state.legit += 1
        st.session_state.fraud_history.append(txn["predicted"])
        st.session_state.fraud_history = st.session_state.fraud_history[-60:]

    # Keep feed to last 100
    st.session_state.transactions = st.session_state.transactions[:100]

# ─────────────────────────────────────────────
# LOAD SAVED MODEL (optional)
# ─────────────────────────────────────────────
@st.cache_resource
def load_saved_model():
    paths = [
        "models/random_forest_model.pkl",
        "models/xgboost_model.pkl",
    ]
    for p in paths:
        if os.path.exists(p):
            return joblib.load(p), os.path.basename(p)
    return None, None

model, model_name = load_saved_model()

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def fmt_inr(v):
    if v >= 1_00_000:
        return f"₹{v/1_00_000:.1f}L"
    if v >= 1000:
        return f"₹{v/1000:.1f}K"
    return f"₹{v:.0f}"

def fraud_rate_pct():
    if st.session_state.total == 0:
        return 0.0
    return round(st.session_state.fraud / st.session_state.total * 100, 1)

def system_risk():
    return min(int(fraud_rate_pct() * 5), 100)

# ─────────────────────────────────────────────
# ── SIDEBAR ──────────────────────────────────
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ FraudShield")
    st.markdown(
        "<div style='font-size:9px;color:#3a6080;letter-spacing:0.15em;'>"
        "SECUREBANK MONITORING SYSTEM v2.4.1</div>",
        unsafe_allow_html=True
    )
    st.divider()

    # Model status
    if model:
        st.success(f"✓ Model loaded: `{model_name}`")
    else:
        st.info("⚠ No saved model found.\nUsing synthetic simulation.\n\nRun `python main.py` first to train & save models.")

    st.divider()
    st.markdown("<div class='section-header'>⚙ SIMULATION CONTROLS</div>", unsafe_allow_html=True)

    speed = st.select_slider(
        "Refresh Speed",
        options=["Slow (3s)", "Medium (1.5s)", "Fast (0.8s)", "Turbo (0.3s)"],
        value="Medium (1.5s)",
    )
    speed_map = {"Slow (3s)": 3.0, "Medium (1.5s)": 1.5, "Fast (0.8s)": 0.8, "Turbo (0.3s)": 0.3}
    refresh_sec = speed_map[speed]

    batch_size = st.slider("Transactions per refresh", 1, 10, 3)
    fraud_rate_slider = st.slider("Simulated Fraud Rate (%)", 5, 40, 18) / 100

    st.divider()
    st.markdown("<div class='section-header'>◈ SYSTEM STATUS</div>", unsafe_allow_html=True)

    risk = system_risk()
    risk_color = "#ff3a3a" if risk > 70 else "#ffaa00" if risk > 40 else "#00ff88"
    risk_label = "HIGH ⚠" if risk > 70 else "MEDIUM" if risk > 40 else "LOW ✓"
    st.markdown(
        f"<div style='margin:8px 0'>"
        f"<div class='kpi-label'>THREAT LEVEL</div>"
        f"<div style='font-size:18px;color:{risk_color};font-family:Orbitron,monospace;font-weight:bold;'>{risk_label}</div>"
        f"<div style='background:#0d2035;border-radius:3px;height:6px;margin-top:6px;'>"
        f"<div style='width:{risk}%;height:100%;background:{risk_color};border-radius:3px;box-shadow:0 0 8px {risk_color};'></div>"
        f"</div><div style='font-size:9px;color:#3a6080;margin-top:3px;'>Risk Index: {risk}/100</div>"
        f"</div>",
        unsafe_allow_html=True
    )

    st.divider()
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("▶ START" if not st.session_state.running else "⏸ PAUSE"):
            st.session_state.running = not st.session_state.running
    with col_s2:
        if st.button("↺ RESET"):
            for k, v in defaults.items():
                st.session_state[k] = v if not isinstance(v, list) else []
            st.rerun()

    st.divider()
    st.markdown(
        "<div style='font-size:9px;color:#1a3050;letter-spacing:0.1em;line-height:1.8;'>"
        "ACS COLLEGE OF ENGINEERING<br>CREDIT CARD FRAUD DETECTION<br>PLACEMENT PROJECT<br>"
        f"SESSION: {datetime.now().strftime('%Y-%m-%d')}</div>",
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────
# ── HEADER ───────────────────────────────────
# ─────────────────────────────────────────────
hcol1, hcol2, hcol3 = st.columns([5, 2, 2])
with hcol1:
    st.markdown(
        "<h2 style='margin:0;padding:0;'>🛡️ SECUREBANK · FRAUDSHIELD</h2>"
        "<div style='font-size:9px;color:#3a6080;letter-spacing:0.2em;margin-top:2px;'>"
        "REAL-TIME TRANSACTION MONITORING SYSTEM · ML ENGINE: RANDOM FOREST + XGBOOST"
        "</div>",
        unsafe_allow_html=True
    )
with hcol2:
    status_color = "#00ff88" if st.session_state.running else "#ff3a3a"
    status_text  = "● MONITORING ACTIVE" if st.session_state.running else "● FEED PAUSED"
    st.markdown(
        f"<div style='margin-top:10px;padding:8px 14px;"
        f"border:1px solid {status_color}44;border-radius:4px;"
        f"background:{status_color}08;font-size:10px;"
        f"color:{status_color};letter-spacing:0.12em;'>{status_text}</div>",
        unsafe_allow_html=True
    )
with hcol3:
    st.markdown(
        f"<div style='margin-top:10px;text-align:right;"
        f"font-size:10px;color:#3a6080;'>"
        f"<div style='color:#00ccff;font-size:13px;font-weight:bold;'>"
        f"{datetime.now().strftime('%H:%M:%S')}</div>"
        f"{datetime.now().strftime('%d %b %Y')}</div>",
        unsafe_allow_html=True
    )

st.divider()

# ─────────────────────────────────────────────
# ── KPI METRICS ──────────────────────────────
# ─────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("◈ TOTAL TXNs",     f"{st.session_state.total:,}")
k2.metric("⚠ FRAUD CAUGHT",   f"{st.session_state.fraud:,}",  delta=f"+{st.session_state.fraud}" if st.session_state.fraud else None, delta_color="inverse")
k3.metric("✓ APPROVED",       f"{st.session_state.legit:,}",  delta=f"+{st.session_state.legit}" if st.session_state.legit else None)
k4.metric("% FRAUD RATE",     f"{fraud_rate_pct()}%",          delta="HIGH" if fraud_rate_pct() > 20 else None, delta_color="inverse")
k5.metric("₹ VOL PROCESSED",  fmt_inr(st.session_state.total_amount))
k6.metric("🔒 BLOCKED VALUE",  fmt_inr(st.session_state.fraud_amount), delta_color="inverse")

st.divider()

# ─────────────────────────────────────────────
# ── TABS ─────────────────────────────────────
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "◈ LIVE FEED",
    "📊 ANALYTICS",
    "🚨 FRAUD ALERTS",
    "🔬 MODEL INFO",
])

# ═══════════════════════════════════════
# TAB 1 — LIVE TRANSACTION FEED
# ═══════════════════════════════════════
with tab1:
    col_feed, col_right = st.columns([3, 1])

    with col_feed:
        st.markdown("<div class='section-header'>◈ LIVE TRANSACTION FEED</div>", unsafe_allow_html=True)

        if not st.session_state.transactions:
            st.markdown(
                "<div style='text-align:center;padding:60px;color:#1a3050;"
                "border:1px dashed #0d2035;border-radius:4px;font-size:12px;"
                "letter-spacing:0.2em;'>▶ PRESS START IN SIDEBAR TO BEGIN MONITORING</div>",
                unsafe_allow_html=True
            )
        else:
            # Build display dataframe
            rows = []
            for t in st.session_state.transactions:
                rows.append({
                    "TIME":     t["timestamp"],
                    "TXN ID":   t["id"],
                    "CARD":     t["card"],
                    "MERCHANT": t["merchant"],
                    "AMOUNT":   f"₹{t['amount']:.2f}",
                    "LOCATION": t["location"],
                    "HOUR":     f"{t['hour']:02d}:00",
                    "RISK %":   f"{t['prob']*100:.0f}%",
                    "STATUS":   t["status"],
                })
            df_display = pd.DataFrame(rows)

            # Color rows
            def style_rows(row):
                if "BLOCKED" in row["STATUS"]:
                    return ["background-color:#100808; color:#ff8888"] * len(row)
                return ["background-color:#070f1a; color:#8ab4d4"] * len(row)

            styled = df_display.style.apply(style_rows, axis=1)
            st.dataframe(
                styled,
                use_container_width=True,
                height=460,
                hide_index=True,
            )

    with col_right:
        st.markdown("<div class='section-header'>⚡ RECENT ALERTS</div>", unsafe_allow_html=True)
        if not st.session_state.alerts:
            st.markdown("<div style='color:#1a3050;font-size:11px;text-align:center;padding:20px 0;'>No alerts yet</div>", unsafe_allow_html=True)
        else:
            for a in st.session_state.alerts[:6]:
                st.markdown(
                    f"<div class='alert-card'>"
                    f"<div style='display:flex;justify-content:space-between;'>"
                    f"<span style='color:#ff8888;font-size:11px;font-weight:bold;'>{a['id']}</span>"
                    f"<span style='color:#ff3a3a;font-size:11px;'>₹{a['amount']:.2f}</span>"
                    f"</div>"
                    f"<div style='font-size:9px;color:#3a6080;margin-top:3px;'>{a['merchant']}</div>"
                    f"<div style='font-size:9px;color:#3a6080;'>{a['location']} · {a['hour']:02d}:00h</div>"
                    f"<div style='font-size:9px;color:#ff3a3a;margin-top:4px;'>"
                    f"Risk: {a['prob']*100:.0f}% · {a['timestamp']}</div>"
                    f"<div style='font-size:8px;color:#ff550088;margin-top:2px;'>▸ {a['triggers']}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

# ═══════════════════════════════════════
# TAB 2 — ANALYTICS
# ═══════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>📊 REAL-TIME ANALYTICS</div>", unsafe_allow_html=True)

    if len(st.session_state.fraud_history) < 2:
        st.info("Start the feed to generate analytics charts.")
    else:
        ac1, ac2 = st.columns(2)

        with ac1:
            # Fraud rate over time (rolling window)
            history = st.session_state.fraud_history
            window  = 10
            rolling_rate = [
                round(sum(history[max(0, i-window):i+1]) / min(i+1, window) * 100, 1)
                for i in range(len(history))
            ]
            fig_rate = go.Figure()
            fig_rate.add_trace(go.Scatter(
                y=rolling_rate,
                mode="lines",
                name="Fraud Rate %",
                line=dict(color="#ff3a3a", width=2),
                fill="tozeroy",
                fillcolor="rgba(255,58,58,0.08)",
            ))
            fig_rate.add_hline(y=15, line_dash="dot", line_color="#ffaa00",
                               annotation_text="THRESHOLD 15%", annotation_font_color="#ffaa00")
            fig_rate.update_layout(
                title="Rolling Fraud Rate (%)",
                **PLOT_DARK,
                yaxis_title="Fraud %",
                xaxis_title="Transaction #",
                showlegend=False,
            )
            st.plotly_chart(fig_rate, use_container_width=True)

        with ac2:
            # Legit vs Fraud donut
            fig_donut = go.Figure(go.Pie(
                labels=["Legitimate", "Fraud"],
                values=[max(st.session_state.legit, 0), max(st.session_state.fraud, 0)],
                hole=0.65,
                marker=dict(colors=["#00ff88", "#ff3a3a"],
                            line=dict(color="#060d18", width=2)),
                textinfo="percent+label",
                textfont=dict(color="#8ab4d4", size=11),
            ))
            fig_donut.update_layout(
                title="Transaction Distribution",
                **PLOT_DARK,
                showlegend=True,
                legend=dict(font=dict(color="#8ab4d4")),
                annotations=[dict(
                    text=f"{fraud_rate_pct()}%<br>FRAUD",
                    x=0.5, y=0.5, font_size=14,
                    showarrow=False,
                    font=dict(color="#ff3a3a", family="Orbitron, monospace"),
                )],
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        ac3, ac4 = st.columns(2)

        with ac3:
            # Amount distribution from live data
            if st.session_state.transactions:
                fraud_amts = [t["amount"] for t in st.session_state.transactions if t["predicted"] == 1]
                legit_amts = [t["amount"] for t in st.session_state.transactions if t["predicted"] == 0]
                fig_hist = go.Figure()
                if legit_amts:
                    fig_hist.add_trace(go.Histogram(
                        x=legit_amts, name="Legitimate",
                        marker_color="rgba(0,255,136,0.5)", nbinsx=25,
                    ))
                if fraud_amts:
                    fig_hist.add_trace(go.Histogram(
                        x=fraud_amts, name="Fraud",
                        marker_color="rgba(255,58,58,0.5)", nbinsx=25,
                    ))
                fig_hist.update_layout(
                    title="Amount Distribution: Fraud vs Legit",
                    barmode="overlay",
                    **PLOT_DARK,
                    xaxis_title="Amount (₹)",
                    yaxis_title="Count",
                )
                st.plotly_chart(fig_hist, use_container_width=True)

        with ac4:
            # Hour-of-day fraud heatmap
            if st.session_state.transactions:
                hour_fraud  = [0] * 24
                hour_total  = [0] * 24
                for t in st.session_state.transactions:
                    h = t["hour"]
                    hour_total[h] += 1
                    if t["predicted"] == 1:
                        hour_fraud[h] += 1
                hour_rate = [
                    round(hour_fraud[h] / hour_total[h] * 100, 1) if hour_total[h] > 0 else 0
                    for h in range(24)
                ]
                fig_hour = go.Figure(go.Bar(
                    x=list(range(24)),
                    y=hour_rate,
                    marker=dict(
                        color=hour_rate,
                        colorscale=[[0, "#00ff8840"], [0.5, "#ffaa0080"], [1, "#ff3a3acc"]],
                        showscale=False,
                    ),
                ))
                fig_hour.update_layout(
                    title="Fraud Rate by Hour of Day",
                    **PLOT_DARK,
                    xaxis_title="Hour",
                    yaxis_title="Fraud Rate %",
                    xaxis=dict(tickvals=list(range(24)), **PLOT_DARK["xaxis"]),
                )
                fig_hour.add_vrect(x0=-0.5, x1=5.5, fillcolor="#ff3a3a08",
                                   line_width=0, annotation_text="High Risk Hours",
                                   annotation_font_color="#ff3a3a", annotation_font_size=9)
                st.plotly_chart(fig_hour, use_container_width=True)

# ═══════════════════════════════════════
# TAB 3 — FRAUD ALERTS
# ═══════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>🚨 FRAUD ALERT CENTRE</div>", unsafe_allow_html=True)

    if not st.session_state.alerts:
        st.markdown(
            "<div style='text-align:center;padding:60px;color:#1a3050;"
            "border:1px dashed #ff3a3a22;border-radius:4px;font-size:12px;"
            "letter-spacing:0.2em;'>NO FRAUD ALERTS YET · SYSTEM MONITORING</div>",
            unsafe_allow_html=True
        )
    else:
        # Summary row
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Total Alerts", len(st.session_state.alerts))
        sc2.metric("Avg Fraud Risk", f"{round(sum(a['prob'] for a in st.session_state.alerts)/len(st.session_state.alerts)*100, 1)}%")
        sc3.metric("Total Blocked", fmt_inr(sum(a['amount'] for a in st.session_state.alerts)))

        st.markdown("---")

        # Alert cards — 2 columns
        alert_c1, alert_c2 = st.columns(2)
        for i, a in enumerate(st.session_state.alerts):
            col = alert_c1 if i % 2 == 0 else alert_c2
            with col:
                risk_pct = int(a["prob"] * 100)
                st.markdown(
                    f"<div class='alert-card'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
                    f"<span style='color:#ff8888;font-size:13px;font-weight:bold;'>🚨 {a['id']}</span>"
                    f"<span style='background:#ff3a3a22;border:1px solid #ff3a3a55;color:#ff3a3a;"
                    f"padding:2px 8px;border-radius:3px;font-size:10px;'>BLOCKED</span>"
                    f"</div>"
                    f"<div style='margin:8px 0;display:grid;grid-template-columns:1fr 1fr;gap:6px;font-size:10px;'>"
                    f"<div><div class='kpi-label'>CARD</div><div style='color:#8ab4d4;'>{a['card']}</div></div>"
                    f"<div><div class='kpi-label'>AMOUNT</div><div style='color:#ffaa00;font-size:14px;'>₹{a['amount']:.2f}</div></div>"
                    f"<div><div class='kpi-label'>MERCHANT</div><div style='color:#8ab4d4;'>{a['merchant']}</div></div>"
                    f"<div><div class='kpi-label'>LOCATION</div><div style='color:#8ab4d4;'>{a['location']}</div></div>"
                    f"<div><div class='kpi-label'>HOUR</div><div style='color:{'#ffaa00' if a['hour'] < 6 else '#8ab4d4'};'>{a['hour']:02d}:00 {'⚠' if a['hour'] < 6 else ''}</div></div>"
                    f"<div><div class='kpi-label'>TIME</div><div style='color:#8ab4d4;'>{a['timestamp']}</div></div>"
                    f"</div>"
                    f"<div style='margin-top:6px;'>"
                    f"<div class='kpi-label'>FRAUD PROBABILITY</div>"
                    f"<div style='display:flex;align-items:center;gap:8px;margin-top:3px;'>"
                    f"<div style='flex:1;height:5px;background:#0d2035;border-radius:3px;'>"
                    f"<div style='width:{risk_pct}%;height:100%;background:#ff3a3a;"
                    f"border-radius:3px;box-shadow:0 0 6px #ff3a3a;'></div></div>"
                    f"<span style='color:#ff3a3a;font-size:12px;font-weight:bold;'>{risk_pct}%</span>"
                    f"</div></div>"
                    f"<div style='margin-top:6px;font-size:9px;color:#ff550099;'>"
                    f"▸ TRIGGERS: {a['triggers']}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

        # Export button
        st.divider()
        alert_df = pd.DataFrame([{
            "TXN ID": a["id"], "Card": a["card"], "Merchant": a["merchant"],
            "Amount (₹)": a["amount"], "Location": a["location"],
            "Hour": a["hour"], "Risk %": round(a["prob"]*100, 1),
            "Triggers": a["triggers"], "Timestamp": a["timestamp"],
        } for a in st.session_state.alerts])

        st.download_button(
            label="⬇ EXPORT FRAUD ALERTS (CSV)",
            data=alert_df.to_csv(index=False),
            file_name=f"fraud_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

# ═══════════════════════════════════════
# TAB 4 — MODEL INFO
# ═══════════════════════════════════════
with tab4:
    st.markdown("<div class='section-header'>🔬 MODEL INFORMATION</div>", unsafe_allow_html=True)

    mi1, mi2 = st.columns(2)
    with mi1:
        st.markdown("#### About This Project")
        st.markdown("""
<div style='font-size:12px;line-height:1.9;color:#8ab4d4;'>

**Dataset:** Kaggle Credit Card Fraud Detection
- 284,807 transactions
- 492 fraudulent (0.17%)
- Features: V1–V28 (PCA), Time, Amount

**Models Trained:**
- `Random Forest` (100 trees, max_depth=10)
- `XGBoost` (gradient boosting, 100 estimators)

**Imbalance Handling:**
- SMOTE applied to training set only
- Balances fraud from 0.17% → 50%

**Key Metrics (typical results):**
| Metric | Random Forest | XGBoost |
|--------|--------------|---------|
| ROC-AUC | ~0.974 | ~0.979 |
| Precision | ~0.87 | ~0.89 |
| Recall | ~0.84 | ~0.83 |

</div>
""", unsafe_allow_html=True)

    with mi2:
        st.markdown("#### Fraud Detection Pipeline")
        st.markdown("""
<div style='font-size:11px;line-height:2.0;color:#8ab4d4;
background:#080f1c;border:1px solid #0d2035;border-radius:4px;padding:16px;
font-family:\"Share Tech Mono\",monospace;'>

<span style='color:#00ccff;'>INPUT</span>   → Transaction Data (Amount, Time, V1-V28)<br>
    ↓<br>
<span style='color:#00ccff;'>CLEAN</span>   → Remove duplicates, null check<br>
    ↓<br>
<span style='color:#00ccff;'>ENGINEER</span> → Log(Amount), Hour extraction, Bins<br>
    ↓<br>
<span style='color:#00ccff;'>SCALE</span>   → StandardScaler on Amount & Time<br>
    ↓<br>
<span style='color:#00ccff;'>SMOTE</span>   → Balance training data<br>
    ↓<br>
<span style='color:#00ccff;'>MODEL</span>   → Random Forest + XGBoost<br>
    ↓<br>
<span style='color:#00ccff;'>SCORE</span>   → Fraud Probability (0.0 → 1.0)<br>
    ↓<br>
<span style='color:#ff3a3a;'>ALERT</span>   → 🚨 BLOCK if score &gt; threshold

</div>
""", unsafe_allow_html=True)

        if model:
            st.success(f"✓ **{model_name}** is loaded and ready for real predictions.")
            st.caption("To use real predictions: modify `generate_transaction()` to pass features through `model.predict_proba()`")
        else:
            st.warning("No trained model found. Run `python main.py` first.")

    st.divider()
    st.markdown(
        "<div style='font-size:9px;color:#1a3050;text-align:center;letter-spacing:0.15em;'>"
        "CREDIT CARD FRAUD DETECTION SYSTEM · ACS COLLEGE OF ENGINEERING, BENGALURU · "
        "VTU · CYBERSECURITY SPECIALISATION · PLACEMENT PROJECT 2025"
        "</div>",
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────
# ── AUTO-REFRESH LOOP ────────────────────────
# ─────────────────────────────────────────────
if st.session_state.running:
    process_batch(batch_size, fraud_rate_slider)
    time.sleep(refresh_sec)
    st.rerun()