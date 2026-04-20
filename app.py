"""
app.py — PNC Retail Banking Performance Monitor
Streamlit executive dashboard: variance analysis, forecasting, KPI cards, export
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from datetime import datetime
import io

# ─────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PNC Retail Banking Monitor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2rem; font-weight: 700; color: #003366;
        border-bottom: 3px solid #003366; padding-bottom: 0.5rem; margin-bottom: 1.5rem;
    }
    .sub-header {
        font-size: 0.95rem; color: #6c757d; margin-top: -1rem; margin-bottom: 2rem;
    }
    .section-title {
        font-size: 1.1rem; font-weight: 700; color: #003366;
        margin-top: 2rem; margin-bottom: 0.8rem; padding-bottom: 0.3rem;
        border-bottom: 2px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
MATERIALITY_THRESHOLD = 5.0
WMA_WEIGHTS = [0.50, 0.30, 0.20]
FORECAST_PERIODS = 2
FORECAST_LABELS = ['Q2 2026 (F)', 'Q3 2026 (F)']

KEY_METRICS = [
    'net_interest_income', 'noninterest_income', 'total_revenue',
    'noninterest_expense', 'provision_for_credit_losses', 'earnings',
    'avg_loans_billions', 'avg_deposits_billions'
]

METRIC_LABELS = {
    'net_interest_income':         'Net Interest Income ($M)',
    'noninterest_income':          'Noninterest Income ($M)',
    'total_revenue':               'Total Revenue ($M)',
    'noninterest_expense':         'Noninterest Expense ($M)',
    'provision_for_credit_losses': 'Provision for Credit Losses ($M)',
    'earnings':                    'Segment Earnings ($M)',
    'avg_loans_billions':          'Avg Loans ($B)',
    'avg_deposits_billions':       'Avg Deposits ($B)',
}

UNFAVORABLE_METRICS = [
    'noninterest_expense',
    'provision_for_credit_losses',
    'net_loan_charge_offs'
]

# ─────────────────────────────────────────────
# EMBEDDED DATA — no CSV file needed
# Source: PNC Quarterly Earnings Releases
# investor.pnc.com — Retail Banking Segment
# All figures in $millions except loans/deposits in $billions
# ─────────────────────────────────────────────
RAW_DATA = {
    'quarter':      ['2025Q1', '2025Q2', '2025Q3', '2026Q1'],
    'period_label': ['Q1 2025', 'Q2 2025', 'Q3 2025', 'Q1 2026'],
    'net_interest_income':         [2826, 2974, 3016, 3198],
    'noninterest_income':          [706,  782,  790,  770],
    'noninterest_expense':         [1903, 1890, 1941, 2115],
    'provision_for_credit_losses': [168,  83,   126,  124],
    'earnings':                    [1112, 1359, 1324, 1320],
    'avg_loans_billions':          [95.6, 97.5, 96.9, 110.9],
    'avg_deposits_billions':       [245.1, 243.5, 243.3, 268.2],
    'net_loan_charge_offs':        [144,  120,  126,  118]
}

# ─────────────────────────────────────────────
# DATA PROCESSING
# ─────────────────────────────────────────────
@st.cache_data
def load_and_process(uploaded_file=None):
    """Load data from upload if provided, otherwise use embedded data."""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.DataFrame(RAW_DATA)

    df = df.sort_values('quarter').reset_index(drop=True)

    financial_cols = [
        'net_interest_income', 'noninterest_income', 'noninterest_expense',
        'provision_for_credit_losses', 'earnings', 'avg_loans_billions',
        'avg_deposits_billions', 'net_loan_charge_offs'
    ]
    for col in financial_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace('[$,]', '', regex=True),
                errors='coerce'
            )

    df['total_revenue'] = df['net_interest_income'] + df['noninterest_income']
    df['efficiency_ratio'] = (
        df['noninterest_expense'] / df['total_revenue'] * 100
    ).round(1)

    for metric in KEY_METRICS:
        if metric in df.columns:
            prior = df[metric].shift(1)
            df[f'{metric}_qoq_chg'] = df[metric] - prior
            df[f'{metric}_qoq_pct'] = (
                (df[metric] - prior) / prior.abs() * 100
            ).round(2)
            prior_y = df[metric].shift(4)
            df[f'{metric}_yoy_chg'] = df[metric] - prior_y
            df[f'{metric}_yoy_pct'] = (
                (df[metric] - prior_y) / prior_y.abs() * 100
            ).round(2)

    return df


@st.cache_data
def build_forecasts(df):
    """Compute weighted moving average forecasts."""
    forecast_metrics = [
        'net_interest_income', 'noninterest_income', 'total_revenue',
        'noninterest_expense', 'earnings', 'avg_loans_billions'
    ]
    results = {}
    for metric in forecast_metrics:
        if metric not in df.columns:
            continue
        actuals = df[metric].dropna().values
        if len(actuals) < 3:
            results[metric] = {
                'forecasts': [np.nan] * FORECAST_PERIODS,
                'upper':     [np.nan] * FORECAST_PERIODS,
                'lower':     [np.nan] * FORECAST_PERIODS
            }
            continue

        last_3   = actuals[-3:]
        wma_base = np.dot(WMA_WEIGHTS, last_3[::-1])

        x = np.arange(len(actuals))
        slope, _, r_value, _, _ = stats.linregress(x, actuals)
        trend   = slope if r_value ** 2 > 0.5 else 0
        std_dev = np.std(actuals)

        forecasts, upper, lower = [], [], []
        for i in range(1, FORECAST_PERIODS + 1):
            val = wma_base + trend * i
            forecasts.append(round(val, 1))
            upper.append(round(val + std_dev, 1))
            lower.append(round(val - std_dev, 1))

        results[metric] = {
            'forecasts': forecasts,
            'upper':     upper,
            'lower':     lower
        }
    return results


def get_flagged_variances(df, threshold=MATERIALITY_THRESHOLD):
    """Return flagged variances for the most recent quarter."""
    latest = df.iloc[-1]
    flags  = []
    for metric in KEY_METRICS:
        if metric not in df.columns:
            continue
        qoq = latest.get(f'{metric}_qoq_pct', np.nan)
        yoy = latest.get(f'{metric}_yoy_pct', np.nan)
        if (pd.notnull(qoq) and abs(qoq) >= threshold) or \
           (pd.notnull(yoy) and abs(yoy) >= threshold):
            label      = METRIC_LABELS.get(metric, metric)
            commentary = []
            if pd.notnull(qoq) and abs(qoq) >= threshold:
                direction = 'increased' if qoq > 0 else 'decreased'
                is_good   = (qoq > 0) if metric not in UNFAVORABLE_METRICS else (qoq < 0)
                icon      = '✅' if is_good else '⚠️'
                commentary.append(
                    f"{icon} **{label}** {direction} "
                    f"**{abs(qoq):.1f}% QoQ** — exceeds ±{threshold}% threshold."
                )
            if pd.notnull(yoy) and abs(yoy) >= threshold:
                direction = 'increased' if yoy > 0 else 'decreased'
                is_good   = (yoy > 0) if metric not in UNFAVORABLE_METRICS else (yoy < 0)
                icon      = '✅' if is_good else '⚠️'
                commentary.append(
                    f"{icon} **{label}** {direction} "
                    f"**{abs(yoy):.1f}% YoY** — seasonality-adjusted signal."
                )
            flags.append({
                'metric':     metric,
                'label':      label,
                'commentary': ' | '.join(commentary)
            })
    return flags


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 PNC Retail Banking")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "📂 Upload New Data CSV (optional)",
        type=['csv'],
        help="Leave empty to use the embedded PNC data. "
             "Upload a CSV with the same column structure to use your own data."
    )

    st.markdown("---")
    st.markdown("**📊 Metric Filters**")
    show_nii      = st.checkbox("Net Interest Income", value=True)
    show_expense  = st.checkbox("Noninterest Expense", value=True)
    show_earnings = st.checkbox("Segment Earnings",    value=True)

    st.markdown("---")
    st.markdown("**🔮 Forecast**")
    show_forecast = st.toggle("Show Forecast", value=True)

    st.markdown("---")
    st.caption(
        "Data: PNC Quarterly Earnings Releases | "
        "investor.pnc.com | Portfolio project"
    )


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
df = load_and_process(uploaded_file)

all_quarters   = df['period_label'].tolist()
forecasts      = build_forecasts(df)
latest         = df.iloc[-1]
prior          = df.iloc[-2]

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown(
    '<div class="main-header">🏦 PNC Retail Banking Performance Monitor</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-header">Real reported figures &nbsp;|&nbsp; '
    'Quarterly variance analysis &nbsp;|&nbsp; '
    'Rolling forecast &nbsp;|&nbsp; Audit-ready export</div>',
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────
st.markdown(
    f'<div class="section-title">📌 Latest Quarter — {latest["period_label"]}</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

def kpi_card(col, label, value, prior_value, fmt='${:,.0f}M', unfavorable=False):
    if prior_value and prior_value != 0:
        delta_pct  = (value - prior_value) / abs(prior_value) * 100
        delta_str  = f"{delta_pct:+.1f}% QoQ"
        delta_color = "normal" if (delta_pct > 0) != unfavorable else "inverse"
    else:
        delta_str   = "N/A"
        delta_color = "off"
    col.metric(
        label=label,
        value=fmt.format(value),
        delta=delta_str,
        delta_color=delta_color
    )

kpi_card(col1, "Net Interest Income", latest['net_interest_income'], prior['net_interest_income'])
kpi_card(col2, "Noninterest Expense", latest['noninterest_expense'], prior['noninterest_expense'], unfavorable=True)
kpi_card(col3, "Segment Earnings",    latest['earnings'],            prior['earnings'])
kpi_card(col4, "Efficiency Ratio",    latest['efficiency_ratio'],    prior['efficiency_ratio'],
         fmt='{:.1f}%', unfavorable=True)

# ─────────────────────────────────────────────
# CHARTS ROW 1: Waterfall + Trend
# ─────────────────────────────────────────────
st.markdown(
    '<div class="section-title">📊 Variance Bridge & Trend Analysis</div>',
    unsafe_allow_html=True
)
chart_col1, chart_col2 = st.columns([1, 1.3])

# --- Waterfall ---
with chart_col1:
    bridge = {
        f'Prior\n({prior["period_label"]})':    prior['earnings'],
        'NII\nChange':      latest['net_interest_income'] - prior['net_interest_income'],
        'Fee\nChange':      latest['noninterest_income']  - prior['noninterest_income'],
        'Expense\nChange': -(latest['noninterest_expense'] - prior['noninterest_expense']),
        'Provision\nChange': -(latest['provision_for_credit_losses'] - prior['provision_for_credit_losses']),
        f'Current\n({latest["period_label"]})': latest['earnings'],
    }
    measure = ['absolute', 'relative', 'relative', 'relative', 'relative', 'total']

    fig_wf = go.Figure(go.Waterfall(
        measure=measure,
        x=list(bridge.keys()),
        y=list(bridge.values()),
        text=[f'${v:+,.0f}M' if i not in [0, 5] else f'${v:,.0f}M'
              for i, v in enumerate(bridge.values())],
        textposition='outside',
        increasing={'marker': {'color': '#28a745'}},
        decreasing={'marker': {'color': '#dc3545'}},
        totals={'marker':    {'color': '#003366'}},
        connector={'line':   {'color': '#adb5bd', 'dash': 'dot', 'width': 1}}
    ))
    fig_wf.update_layout(
        title='QoQ Earnings Bridge',
        yaxis_title='$ Millions',
        template='plotly_white',
        height=400,
        showlegend=False,
        margin=dict(l=50, r=20, t=50, b=60)
    )
    st.plotly_chart(fig_wf, use_container_width=True)

# --- Trend Lines ---
with chart_col2:
    actuals_labels = df['period_label'].tolist()
    fig_trend      = go.Figure()

    trend_cfg = []
    if show_nii:
        trend_cfg.append(('net_interest_income', 'Net Interest Income', '#003366'))
    if show_expense:
        trend_cfg.append(('noninterest_expense', 'Noninterest Expense', '#dc3545'))
    if show_earnings:
        trend_cfg.append(('earnings', 'Segment Earnings', '#28a745'))

    for col_name, label, color in trend_cfg:
        fig_trend.add_trace(go.Scatter(
            x=actuals_labels,
            y=df[col_name].tolist(),
            mode='lines+markers',
            name=label,
            line=dict(color=color, width=2.5),
            marker=dict(size=8)
        ))
        if show_forecast and col_name in forecasts:
            bridge_x = [actuals_labels[-1]] + FORECAST_LABELS
            bridge_y = [df[col_name].iloc[-1]] + forecasts[col_name]['forecasts']
            fig_trend.add_trace(go.Scatter(
                x=bridge_x, y=bridge_y,
                mode='lines+markers',
                name=f'{label} (F)',
                line=dict(color=color, width=2, dash='dot'),
                marker=dict(size=7, symbol='diamond')
            ))
            band_x = FORECAST_LABELS + FORECAST_LABELS[::-1]
            band_y = forecasts[col_name]['upper'] + forecasts[col_name]['lower'][::-1]
            fig_trend.add_trace(go.Scatter(
                x=band_x, y=band_y,
                fill='toself',
                fillcolor='rgba(128,128,128,0.10)',
                line=dict(width=0),
                showlegend=False
            ))

    # Manual vertical separator line
    all_y = []
    for col_name, _, _ in trend_cfg:
        all_y += df[col_name].tolist()
    if all_y:
        y_min = min(all_y) * 0.85
        y_max = max(all_y) * 1.10
        fig_trend.add_trace(go.Scatter(
            x=[actuals_labels[-1], actuals_labels[-1]],
            y=[y_min, y_max],
            mode='lines',
            line=dict(color='#6c757d', width=1.5, dash='dash'),
            showlegend=False
        ))
        fig_trend.add_annotation(
            x=actuals_labels[-1],
            y=y_max * 0.97,
            text='← Actuals | Forecast →',
            showarrow=False,
            font=dict(size=10, color='#6c757d'),
            bgcolor='rgba(255,255,255,0.7)'
        )

    fig_trend.update_layout(
        title='Trend & Rolling Forecast',
        yaxis_title='$ Millions',
        template='plotly_white',
        height=400,
        legend=dict(orientation='h', y=-0.25, x=0),
        margin=dict(l=50, r=20, t=50, b=90)
    )
    st.plotly_chart(fig_trend, use_container_width=True)

# ─────────────────────────────────────────────
# HEATMAP
# ─────────────────────────────────────────────
st.markdown(
    '<div class="section-title">🌡️ Variance Heatmap — QoQ % Change</div>',
    unsafe_allow_html=True
)

heatmap_cols = [
    ('net_interest_income',         'Net Interest Income'),
    ('noninterest_income',          'Noninterest Income'),
    ('total_revenue',               'Total Revenue'),
    ('noninterest_expense',         'Noninterest Expense (↑bad)'),
    ('provision_for_credit_losses', 'Provision (↑bad)'),
    ('earnings',                    'Segment Earnings'),
    ('avg_loans_billions',          'Avg Loans ($B)'),
]

z_vals, text_vals, y_labels = [], [], []
x_labels = df['period_label'].tolist()

for col_name, label in heatmap_cols:
    pct_col  = f'{col_name}_qoq_pct'
    row_z, row_text = [], []
    for idx in range(len(df)):
        val = df[pct_col].iloc[idx] if pct_col in df.columns else np.nan
        display_val = (
            -val if col_name in UNFAVORABLE_METRICS and pd.notnull(val) else val
        )
        row_z.append(display_val if pd.notnull(display_val) else 0)
        row_text.append(f'{val:+.1f}%' if pd.notnull(val) else 'N/A')
    z_vals.append(row_z)
    text_vals.append(row_text)
    y_labels.append(label)

fig_heat = go.Figure(data=go.Heatmap(
    z=z_vals, x=x_labels, y=y_labels,
    text=text_vals,
    texttemplate='%{text}',
    textfont=dict(size=12, family='Arial'),
    colorscale=[
        [0.0,  '#dc3545'],
        [0.35, '#fff3cd'],
        [0.5,  '#ffffff'],
        [0.65, '#d4edda'],
        [1.0,  '#155724']
    ],
    zmid=0,
    colorbar=dict(title='Favorable %', ticksuffix='%')
))
fig_heat.update_layout(
    template='plotly_white',
    height=320,
    margin=dict(l=220, r=100, t=30, b=50)
)
st.plotly_chart(fig_heat, use_container_width=True)

# ─────────────────────────────────────────────
# FLAGGED VARIANCES
# ─────────────────────────────────────────────
st.markdown(
    f'<div class="section-title">🚩 Flagged Variances — '
    f'{latest["period_label"]} (≥±{MATERIALITY_THRESHOLD}% threshold)</div>',
    unsafe_allow_html=True
)

flagged = get_flagged_variances(df)
if flagged:
    for flag in flagged:
        st.markdown(flag['commentary'])
        st.markdown("---")
else:
    st.success("✅ No variances exceed the ±5% materiality threshold this quarter.")

# ─────────────────────────────────────────────
# DOWNLOAD BUTTON
# ─────────────────────────────────────────────
st.markdown(
    '<div class="section-title">📥 Audit-Ready Export</div>',
    unsafe_allow_html=True
)

export_cols = (
    ['period_label', 'quarter'] +
    [m for m in KEY_METRICS if m in df.columns] +
    ['total_revenue', 'efficiency_ratio'] +
    [f'{m}_qoq_pct' for m in KEY_METRICS if f'{m}_qoq_pct' in df.columns] +
    [f'{m}_yoy_pct' for m in KEY_METRICS if f'{m}_yoy_pct' in df.columns]
)
export_df = df[[c for c in export_cols if c in df.columns]].copy()

timestamp  = datetime.now().strftime('%Y%m%d_%H%M')
csv_buffer = io.StringIO()
export_df.to_csv(csv_buffer, index=False)
csv_bytes  = csv_buffer.getvalue().encode()

dl_col1, dl_col2 = st.columns([1, 3])
with dl_col1:
    st.download_button(
        label="⬇️ Download Report (CSV)",
        data=csv_bytes,
        file_name=f'PNC_Retail_Variance_{timestamp}.csv',
        mime='text/csv'
    )
with dl_col2:
    st.caption(
        f"Includes actuals, QoQ/YoY variances, and efficiency ratio. "
        f"Timestamped: {timestamp}."
    )

st.markdown("---")
st.caption(
    "Data: PNC Financial Services Group Quarterly Earnings Releases "
    "(investor.pnc.com) | Portfolio project | Not investment advice."
)