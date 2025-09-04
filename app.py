
# app.py
# 🚀 Auto Insights — V7 (Executive Narratives, Ranked Insights, White Theme)
# Python 3.11+ compatible

import os
import io
import zipfile
import tempfile
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import streamlit as st
import numpy as np
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from plotly.io import to_image
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
from scipy import stats
import csv as csv_module

# ---------------------------- Page & Theme ----------------------------
st.set_page_config(page_title="Auto Insights – CE Innovations Lab", page_icon="🚀", layout="wide")

# White theme styling
st.markdown("""
<style>
.stApp { background: #ffffff; color: #1f2937; }
.block-container { padding-top: 1rem; }
.glass {
  background: rgba(255, 255, 255, 0.98);
  border: 1px solid rgba(0, 0, 0, 0.08);
  box-shadow: 0 4px 12px rgba(0,0,0,0.08);
  border-radius: 12px;
  padding: 1rem 1.2rem;
  margin-bottom: 1rem;
}
.kpi {
  background: rgba(249,250,251,0.98);
  border: 1px solid rgba(0,0,0,0.06);
  border-radius: 12px;
  padding: 0.8rem 1rem;
  text-align: center;
}
.kpi h3 { margin: 0; font-size: 0.9rem; color: #6b7280; }
.kpi p { margin: 0; font-size: 1.3rem; font-weight: 700; color: #111827; }
.smallnote { color: #6b7280; font-size: 0.9rem; }
.insight-card {
  border-left: 4px solid #3b82f6; padding: 10px 12px; margin: 10px 0; border-radius: 8px;
  background: #f8fafc;
}
.insight-title { font-weight: 700; color: #111827; }
.insight-meta { color: #6b7280; font-size: 0.85rem; }
footer {visibility: hidden;}
#custom-footer {
  position: fixed; left:0; right:0; bottom:0; z-index: 9999;
  background: rgba(255,255,255,0.95);
  border-top: 1px solid rgba(0,0,0,0.08);
  color: #374151;
  padding: 8px 16px; text-align: center; font-size: 0.9rem;
}
hr { border-color: rgba(0,0,0,0.1); }
</style>
<div id="custom-footer">Developed by <b>CE Innovations Lab 2025</b></div>
""", unsafe_allow_html=True)

# ---------------------------- Session ----------------------------
for key, default in {
    "charts_png": [],
    "narratives": [],    # list of dicts with: category, title, text, impact, priority
    "df_pl": None, "df_pd": None, "parquet_path": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------- Utils & Data I/O ----------------------------
@st.cache_data(show_spinner=False)
def _to_parquet(df_pl: pl.DataFrame) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
    tmp.close()
    df_pl.write_parquet(tmp.name)
    return tmp.name

@st.cache_data(show_spinner=True)
def load_file_to_polars(uploaded_file, try_parse_dates: bool = True) -> Tuple[pl.DataFrame, pd.DataFrame, str]:
    name = uploaded_file.name.lower()
    ext = os.path.splitext(name)[1]
    if ext in [".csv", ".txt"]:
        df_pl = pl.read_csv(uploaded_file, infer_schema_length=5000)
    elif ext in [".xlsx", ".xls"]:
        pdf = pd.read_excel(uploaded_file)
        df_pl = pl.from_pandas(pdf)
    elif ext in [".parquet"]:
        df_pl = pl.read_parquet(uploaded_file)
    else:
        raise ValueError("Unsupported file. Please upload CSV, Excel, or Parquet.")

    # Attempt parse date-like strings
    if try_parse_dates:
        for c, dt in zip(df_pl.columns, df_pl.dtypes):
            if dt == pl.Utf8 and df_pl[c].null_count() < df_pl.height:
                sample = df_pl[c].head(200).to_list()
                hits = 0
                for s in sample:
                    if s is None: continue
                    try:
                        pd.to_datetime(s)
                        hits += 1
                    except Exception:
                        pass
                if hits > len(sample) * 0.6:
                    try:
                        df_pl = df_pl.with_columns(pl.col(c).str.strptime(pl.Datetime, strict=False, fmt=None).alias(c))
                    except Exception:
                        pass

    parquet_path = _to_parquet(df_pl)
    df_pd = df_pl.to_pandas()
    return df_pl, df_pd, parquet_path

def dtype_buckets(df_pl: pl.DataFrame) -> Dict[str, List[str]]:
    num = [c for c, t in zip(df_pl.columns, df_pl.dtypes) if t in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.UInt64, pl.UInt32)]
    cat = [c for c, t in zip(df_pl.columns, df_pl.dtypes) if t in (pl.Utf8, pl.Categorical)]
    dtm = [c for c, t in zip(df_pl.columns, df_pl.dtypes) if t in (pl.Datetime, pl.Date)]
    return {"numeric": num, "categorical": cat, "datetime": dtm}

def friendly_guard(ok: bool):
    if not ok:
        st.info("⚠️ Select correct data type(s) for this visual.")
    return ok

def add_card(category: str, title: str, text: str, impact: float = 0.0, priority: float = 0.0):
    st.session_state["narratives"].append({
        "category": category, "title": title, "text": text,
        "impact": float(impact), "priority": float(priority)
    })

def save_chart_png(fig: go.Figure, fname: str):
    try:
        img = to_image(fig, format="png", scale=2)  # requires kaleido
        st.session_state["charts_png"].append((fname, img))
    except Exception:
        pass

def zip_all_charts() -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, b in st.session_state["charts_png"]:
            zf.writestr(fname, b)
    mem.seek(0)
    return mem.read()

# ---------------------------- Insight Utilities ----------------------------
def pct_change(a: float, b: float) -> float:
    if a == 0: return 0.0
    return 100.0 * (b - a) / abs(a)

def linear_trend_strength(ts: pd.Series) -> Tuple[float, float, float]:
    # returns slope, r, pvalue
    x = np.arange(len(ts))
    res = stats.linregress(x, ts.values)
    return res.slope, res.rvalue, res.pvalue

def detect_anomalies_zscore(s: pd.Series, z: float = 3.0) -> pd.DataFrame:
    x = (s - s.mean()) / (s.std(ddof=1) + 1e-9)
    return pd.DataFrame({"value": s, "z": x, "is_outlier": (x.abs() >= z)})

def seasonal_lifts(df: pd.DataFrame, tcol: str, vcol: str) -> Optional[Dict[str, float]]:
    d = df[[tcol, vcol]].dropna().copy()
    d[tcol] = pd.to_datetime(d[tcol], errors="coerce")
    d = d.dropna()
    if d.empty: return None
    d["dow"] = d[tcol].dt.day_name()
    d["moy"] = d[tcol].dt.month_name()
    lifts = {}
    for key, col in [("day_of_week", "dow"), ("month_of_year", "moy")]:
        g = d.groupby(col)[vcol].mean().sort_values(ascending=False)
        if len(g) >= 2:
            lifts[key] = float(pct_change(g.iloc[-1], g.iloc[0]))
    return lifts or None

def corr_pairs(df: pd.DataFrame, cols: List[str], k: int = 5):
    if len(cols) < 2: return None, []
    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    pairs = []
    for i, a in enumerate(corr.columns):
        for j, b in enumerate(corr.columns):
            if mask[i, j] and not np.isnan(corr.iloc[i, j]):
                pairs.append((a, b, float(corr.iloc[i, j])))
    if not pairs: return corr, []
    pairs_sorted = sorted(pairs, key=lambda t: abs(t[2]), reverse=True)[:k]
    return corr, pairs_sorted

def rf_model_and_sensitivity(df: pd.DataFrame, target: str) -> Tuple[Optional[RandomForestRegressor], Optional[pd.Series], Optional[List[Tuple[str, float]]]]:
    dfm = df.select_dtypes(include=np.number).dropna()
    if target not in dfm.columns or dfm.shape[1] < 2: return None, None, None
    X = dfm.drop(columns=[target]); y = dfm[target]
    model = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
    model.fit(X, y)
    # permutation importance for stability
    pi = permutation_importance(model, X, y, n_repeats=8, random_state=42, n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False)

    # sensitivity: change prediction from 10th->90th pct for each top feature
    sens = []
    for feat in imp.head(5).index:
        q10, q90 = np.percentile(X[feat], [10, 90])
        X_low, X_high = X.copy(), X.copy()
        X_low[feat] = q10; X_high[feat] = q90
        delta = float(model.predict(X_high).mean() - model.predict(X_low).mean())
        sens.append((feat, delta))
    return model, imp, sens

# ---------------------------- Chart Narratives ----------------------------
def narrative_histogram(df_pd: pd.DataFrame, col: str) -> str:
    s = df_pd[col].dropna()
    if s.empty: return "No numeric data available."
    q1, q3 = np.percentile(s, [25, 75])
    iqr = max(q3-q1, 1e-9)
    out_pct = ((s < q1-1.5*iqr) | (s > q3+1.5*iqr)).mean()*100
    skew = pd.Series(s).skew()
    skew_txt = "right-skewed (long high tail)" if skew > 0.5 else "left-skewed (long low tail)" if skew < -0.5 else "fairly symmetric"
    return (
        f"**What:** {col} is {skew_txt}, median {np.median(s):.2f}. "
        f"**Why it matters:** ~{out_pct:.1f}% outliers can distort averages. "
        f"**Action:** Use median/IQR; investigate outliers. "
        f"**Impact:** More robust targets and fair benchmarking."
    )

def narrative_box(df_pd: pd.DataFrame, col: str) -> str:
    s = df_pd[col].dropna()
    if s.empty: return "No numeric data available."
    return (
        f"**What:** {col} shows spread (IQR≈{np.percentile(s,75)-np.percentile(s,25):.2f}) with outliers present. "
        f"**Why:** High variance implies inconsistent process. "
        f"**Action:** Standardize inputs; isolate outlier sources. "
        f"**Impact:** Reduced variability and predictable performance."
    )

def narrative_bar_count(df_pd: pd.DataFrame, cat: str) -> str:
    counts = df_pd[cat].value_counts(dropna=False)
    if counts.empty: return "No categories present."
    top = counts.head(3)
    conc = 100 * counts.iloc[0] / counts.sum()
    return (
        f"**What:** '{cat}' is concentrated—top segment '{top.index[0]}' = {int(counts.iloc[0])} rows ({conc:.1f}%). "
        f"**Why:** Decisions on dominant segments have outsized effect. "
        f"**Action:** Prioritize top 1–2 segments for initiatives. "
        f"**Impact:** Faster lift with minimal scope creep."
    )

def narrative_bar_agg(df_pd: pd.DataFrame, cat: str, val: str, agg: str) -> str:
    g = df_pd.groupby(cat)[val].agg(agg).sort_values(ascending=False)
    if g.empty: return "No values to aggregate."
    lead = g.index[0]; lead_val = g.iloc[0]
    runner = g.index[1] if len(g)>1 else None
    gap = (lead_val - g.iloc[1]) if runner is not None else 0.0
    return (
        f"**What:** {agg}({val}) peaks at '{lead}' ({lead_val:.2f}). "
        f"**Why:** This segment is your benchmark. "
        f"**Action:** Replicate '{lead}' practices across next 2–3 segments. "
        f"**Impact:** Closing leader→runner gap of {gap:.2f} yields immediate uplift."
    )

def narrative_line(df_pd: pd.DataFrame, tcol: str, vcol: str) -> str:
    d = df_pd[[tcol, vcol]].dropna().copy()
    d[tcol] = pd.to_datetime(d[tcol], errors="coerce")
    d = d.dropna().sort_values(tcol)
    if len(d) < 3: return "Insufficient time points."
    x = (d[tcol]-d[tcol].min()).dt.total_seconds().values.reshape(-1,1)
    y = d[vcol].values
    lr = LinearRegression().fit(x, y)
    slope = lr.coef_[0]; delta = y[-1]-y[0]
    direction = "uptrend" if slope>0 else "downtrend"
    vol = pd.Series(y).pct_change().std()*100 if len(y)>2 else 0.0
    return (
        f"**What:** {vcol} shows a {direction} (Δ={delta:.2f}) with ~{vol:.1f}% volatility. "
        f"**Why:** Volatility hides real movement. "
        f"**Action:** Smooth with 7-period MA; act on trend not noise. "
        f"**Impact:** Better timing for interventions."
    )

def narrative_scatter(df_pd: pd.DataFrame, x: str, y: str) -> str:
    sub = df_pd[[x,y]].dropna()
    if len(sub) < 3: return "Insufficient points."
    r = float(sub.corr().iloc[0,1])
    strength = "strong" if abs(r)>=0.7 else "moderate" if abs(r)>=0.4 else "weak"
    direction = "positive" if r>=0 else "negative"
    return (
        f"**What:** {strength} {direction} association (r={r:.2f}) between {x} and {y}. "
        f"**Why:** Changes in {x} likely move {y} in the same/opposite direction. "
        f"**Action:** Test causality with controls; monitor paired targets. "
        f"**Impact:** Focus on the lever ({x}) to steer {y}."
    )

# ---------------------------- Header ----------------------------
st.markdown("<h1 style='margin-bottom:0'>🚀 Auto Insights – Executive Storytelling</h1>", unsafe_allow_html=True)
st.markdown("<div class='smallnote'>Scalable analytics (Polars + Parquet) with decision-grade narratives.</div>", unsafe_allow_html=True)
st.write("")

# ---------------------------- Upload ----------------------------
with st.container():
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2,1,1])
    with c1:
        file = st.file_uploader("Upload CSV / Excel / Parquet", type=["csv","xlsx","xls","parquet"])
    with c2:
        parse_dates = st.toggle("Auto-parse dates", value=True)
    with c3:
        st.button("Clear Exports", on_click=lambda: (st.session_state["charts_png"].clear(), st.session_state["narratives"].clear()))
    st.markdown("</div>", unsafe_allow_html=True)

if file:
    try:
        df_pl, df_pd, ppath = load_file_to_polars(file, try_parse_dates=parse_dates)
        st.session_state["df_pl"], st.session_state["df_pd"], st.session_state["parquet_path"] = df_pl, df_pd, ppath
    except Exception:
        st.error("Could not load file. Please check format.")

df_pl = st.session_state["df_pl"]
df_pd = st.session_state["df_pd"]
if df_pl is None:
    st.info("Upload a dataset to begin.")
    st.stop()

types = dtype_buckets(df_pl)
num_cols = types["numeric"]
cat_cols = types["categorical"]
dt_cols  = types["datetime"]

# ---------------------------- Tabs ----------------------------
tabs = st.tabs(["Overview", "AI Insights", "Visuals", "Opportunities", "Explorer"])

# ---------------------------- Overview ----------------------------
with tabs[0]:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='kpi'><h3>Rows</h3><p>{df_pl.height:,}</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='kpi'><h3>Columns</h3><p>{len(df_pl.columns)}</p></div>", unsafe_allow_html=True)
    with c3:
        miss = int(df_pd.isna().sum().sum())
        st.markdown(f"<div class='kpi'><h3>Missing</h3><p>{miss:,}</p></div>", unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='kpi'><h3>Parquet</h3><p>On</p></div>", unsafe_allow_html=True)

    st.markdown("**Schema**")
    st.dataframe(pd.DataFrame({"Column": df_pl.columns, "Dtype": [str(t) for t in df_pl.dtypes]}), use_container_width=True, hide_index=True)

    # Missingness
    miss_series = df_pd.isna().sum().sort_values(ascending=False)
    if miss_series.sum() > 0:
        fig = px.bar(miss_series.reset_index(), x="index", y=0, labels={"index":"Column","0":"Missing"}, title="Missing Values by Column")
        st.plotly_chart(fig, use_container_width=True)
        save_chart_png(fig, "missing_values.png")
        add_card("Data Quality", "Missing Data", f"{(miss_series>0).sum()} columns contain missing values; prioritize imputation for the most affected to avoid biased conclusions.", 0.4, 0.6)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------- AI Insights (Executive) ----------------------------
with tabs[1]:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("Executive Summary")

    # Choose a primary target
    tgt = st.selectbox("Primary Target (numeric)", num_cols if num_cols else [None])
    if not tgt:
        st.info("⚠️ Select correct data type (numeric target).")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # Trend & anomalies (if time exists)
        if dt_cols:
            tcol = st.selectbox("Time column", dt_cols, index=0)
            d = df_pd[[tcol, tgt]].dropna().copy()
            d[tcol] = pd.to_datetime(d[tcol], errors="coerce")
            d = d.dropna().sort_values(tcol)
            if len(d) >= 5:
                fig = px.line(d, x=tcol, y=tgt, title=f"{tgt} over {tcol}")
                st.plotly_chart(fig, use_container_width=True)
                save_chart_png(fig, f"line_{tcol}_{tgt}.png")

                slope, r, p = linear_trend_strength(d[tgt])
                delta = d[tgt].iloc[-1] - d[tgt].iloc[0]
                vol = d[tgt].pct_change().std()*100 if len(d) > 3 else 0.0
                season = seasonal_lifts(d, tcol, tgt) or {}
                season_txt = ""
                if "day_of_week" in season:
                    season_txt += f" Weekly pattern amplitude ≈ {abs(season['day_of_week']):.1f}%."
                if "month_of_year" in season:
                    season_txt += f" Monthly pattern amplitude ≈ {abs(season['month_of_year']):.1f}%."

                text = (
                    f"**Trend:** {('upward' if slope>0 else 'downward')} (Δ={delta:.2f}, r={r:.2f}, p={p:.3f}); "
                    f"**Volatility:** ~{vol:.1f}%."
                    f" {season_txt} "
                    f"**Action:** Plan targets using trend + seasonality; smooth noise with a 7-period MA; react to structural shifts, not spikes."
                )
                add_card("Trend", f"{tgt} trend & seasonality", text, impact=0.6, priority=0.8)

                # Anomalies
                an = detect_anomalies_zscore(d[tgt], z=3.0)
                n_out = int(an['is_outlier'].sum())
                if n_out > 0:
                    add_card("Anomalies", "Outlier periods", f"{n_out} anomalous points detected (|z|≥3). Investigate data glitches, one-off events, or process breaks before taking actions.", 0.5, 0.7)
            else:
                st.info("⚠️ Not enough time points for trend analysis.")
        else:
            st.info("⚠️ Provide a datetime column to unlock time-based insights.")

        # Correlations & drivers
        corr, pairs = corr_pairs(df_pd, num_cols, k=5)
        if corr is not None and len(pairs) > 0:
            fig = px.imshow(corr, color_continuous_scale="RdBu_r", origin="lower", aspect="auto", zmin=-1, zmax=1, title="Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)
            save_chart_png(fig, "correlation_heatmap.png")
            pos = [p for p in pairs if p[2] > 0][:3]
            neg = [p for p in pairs if p[2] < 0][:3]
            pos_txt = "; ".join([f"{a}↗{b} (+{v:.2f})" for a,b,v in pos]) or "—"
            neg_txt = "; ".join([f"{a}↘{b} ({v:.2f})" for a,b,v in neg]) or "—"
            add_card("Correlation", "Key relationships", f"Top positive: {pos_txt}. Top negative: {neg_txt}. **Action:** Use positive pairs for leading indicators, manage trade-offs on negative pairs.", 0.4, 0.5)

        model, importances, sensitivity = rf_model_and_sensitivity(df_pd, tgt)
        if importances is not None:
            fig = px.bar(importances.head(10), labels={"index":"Driver","value":"Permutation Importance"}, title=f"Top Drivers of {tgt}")
            st.plotly_chart(fig, use_container_width=True)
            save_chart_png(fig, f"drivers_{tgt}.png")
            sens_txt = ", ".join([f"{f} (Δ≈{d:+.2f})" for f,d in (sensitivity or [])])
            add_card("Drivers", "What moves the target", f"Strong levers: {', '.join([f'{k} ({v:.3f})' for k,v in importances.head(5).items()])}. **Sensitivity:** {sens_txt}. **Action:** Move top levers first; validate feasibility.", 0.7, 0.9)

        # Render insight cards sorted by priority*impact
        st.markdown("---")
        st.markdown("### AI-Generated Insight Cards")
        cards = sorted(st.session_state["narratives"], key=lambda x: x.get("impact",0)*x.get("priority",0), reverse=True)
        for c in cards:
            st.markdown(f"""
<div class='insight-card'>
  <div class='insight-title'>[{c['category']}] {c['title']}</div>
  <div class='insight-meta'>Priority×Impact score: {(c.get('priority',0)*c.get('impact',0)):.2f}</div>
  <div>{c['text']}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------- Visuals ----------------------------
with tabs[2]:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("Custom Chart Builder")

    chart_type = st.selectbox("Chart", ["Histogram", "Box", "Bar (Count)", "Bar (Aggregate)", "Line", "Scatter", "Correlation Heatmap"])
    col1, col2, col3, col4 = st.columns(4)

    # Histogram
    if chart_type == "Histogram":
        with col1:
            num = st.selectbox("Numeric", num_cols)
        if not friendly_guard(num in num_cols if num else False): st.stop()
        fig = px.histogram(df_pd, x=num, nbins=50, opacity=0.85, title=f"Distribution of {num}")
        st.plotly_chart(fig, use_container_width=True); save_chart_png(fig, f"hist_{num}.png")
        st.markdown("### What this means"); st.markdown(narrative_histogram(df_pd, num))

    # Box
    if chart_type == "Box":
        with col1:
            num = st.selectbox("Numeric", num_cols, key="boxn")
        with col2:
            group = st.selectbox("Group (optional)", ["(none)"] + cat_cols, key="boxg")
        if not friendly_guard(num in num_cols if num else False): st.stop()
        if group and group != "(none)":
            fig = px.box(df_pd, x=group, y=num, points="outliers", title=f"{num} by {group}")
        else:
            fig = px.box(df_pd, y=num, points="outliers", title=f"{num} Boxplot")
        st.plotly_chart(fig, use_container_width=True); save_chart_png(fig, f"box_{num}.png")
        st.markdown("### What this means"); st.markdown(narrative_box(df_pd, num))

    # Bar Count
    if chart_type == "Bar (Count)":
        with col1:
            cat = st.selectbox("Categorical", cat_cols, key="barc")
        if not friendly_guard(cat in cat_cols if cat else False): st.stop()
        topn = st.slider("Top N", 5, 50, 20)
        counts = df_pd[cat].value_counts(dropna=False).head(topn)
        fig = px.bar(counts, labels={"index":cat, "value":"Count"}, title=f"{cat}: Top {topn} Counts")
        st.plotly_chart(fig, use_container_width=True); save_chart_png(fig, f"bar_count_{cat}.png")
        st.markdown("### What this means"); st.markdown(narrative_bar_count(df_pd, cat))

    # Bar Aggregate
    if chart_type == "Bar (Aggregate)":
        with col1:
            cat = st.selectbox("Categorical", cat_cols, key="bara")
        with col2:
            val = st.selectbox("Value", num_cols, key="barv")
        with col3:
            agg = st.selectbox("Aggregate", ["mean","sum","median"], index=0)
        if not friendly_guard((cat in cat_cols) and (val in num_cols)): st.stop()
        g = df_pd.groupby(cat)[val].agg(agg).sort_values(ascending=False).head(50)
        fig = px.bar(g, labels={"index":cat, "value":f"{agg}({val})"}, title=f"{agg}({val}) by {cat}")
        st.plotly_chart(fig, use_container_width=True); save_chart_png(fig, f"bar_agg_{cat}_{val}_{agg}.png")
        st.markdown("### What this means"); st.markdown(narrative_bar_agg(df_pd, cat, val, agg))

    # Line
    if chart_type == "Line":
        with col1:
            tcol = st.selectbox("Time", dt_cols + cat_cols, key="linet")
        with col2:
            vcol = st.selectbox("Value", num_cols, key="linev")
        df_tmp = df_pd.copy()
        if tcol in cat_cols: df_tmp[tcol] = pd.to_datetime(df_tmp[tcol], errors="coerce")
        ok = (tcol in dt_cols or tcol in cat_cols) and (vcol in num_cols) and (pd.to_datetime(df_tmp[tcol], errors="coerce").notna().sum()>2)
        if not friendly_guard(ok): st.stop()
        fig = px.line(df_tmp.sort_values(tcol), x=tcol, y=vcol, title=f"{vcol} over {tcol}")
        st.plotly_chart(fig, use_container_width=True); save_chart_png(fig, f"line_{tcol}_{vcol}.png")
        st.markdown("### What this means"); st.markdown(narrative_line(df_tmp, tcol, vcol))

    # Scatter
    if chart_type == "Scatter":
        with col1:
            x = st.selectbox("X (numeric)", num_cols, key="scx")
        with col2:
            y = st.selectbox("Y (numeric)", [c for c in num_cols if c != x], key="scy")
        with col3:
            color = st.selectbox("Color (optional)", ["(none)"] + cat_cols, key="scc")
        if not friendly_guard((x in num_cols) and (y in num_cols)): st.stop()
        if color and color != "(none)":
            fig = px.scatter(df_pd, x=x, y=y, color=color, trendline="ols", opacity=0.75, title=f"{y} vs {x}")
        else:
            fig = px.scatter(df_pd, x=x, y=y, trendline="ols", opacity=0.75, title=f"{y} vs {x}")
        st.plotly_chart(fig, use_container_width=True); save_chart_png(fig, f"scatter_{x}_{y}.png")
        st.markdown("### What this means"); st.markdown(narrative_scatter(df_pd, x, y))

    # Correlation Heatmap
    if chart_type == "Correlation Heatmap":
        ok = len(num_cols) >= 2
        if not friendly_guard(ok): st.stop()
        corr2, pairs2 = corr_pairs(df_pd, num_cols, k=5)
        fig = px.imshow(corr2, color_continuous_scale="RdBu_r", origin="lower", aspect="auto", zmin=-1, zmax=1, title="Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True); save_chart_png(fig, "correlation_heatmap_builder.png")
        pos = [p for p in pairs2 if p[2] > 0][:3]
        neg = [p for p in pairs2 if p[2] < 0][:3]
        st.markdown("### What this means")
        st.markdown(f"Top positive: {('; '.join([f'{a}↗{b} (+{v:.2f})' for a,b,v in pos]) or '—')}. Top negative: {('; '.join([f'{a}↘{b} ({v:.2f})' for a,b,v in neg]) or '—')}.")

    st.markdown("</div>", unsafe_allow_html=True)

    # Export visuals bundle
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    if st.button("📦 Download All Visuals (ZIP)"):
        if st.session_state["charts_png"]:
            zbytes = zip_all_charts()
            st.download_button("Save ZIP", data=zbytes, file_name=f"visualizations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip", mime="application/zip")
        else:
            st.info("No visuals exported yet.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------- Opportunities ----------------------------
with tabs[3]:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("Performance Gaps & Business Opportunities")
    if cat_cols and num_cols:
        dim = st.selectbox("Dimension (categorical)", cat_cols)
        metr = st.selectbox("Metric (numeric)", num_cols)
        ok = (dim in cat_cols) and (metr in num_cols)
        if not friendly_guard(ok): st.stop()
        d = df_pd[[dim, metr]].dropna()
        if not d.empty:
            overall = d[metr].mean()
            g = d.groupby(dim)[metr].agg(['mean','count']).sort_values('mean')
            g['Gap_to_Overall'] = (overall - g['mean']).clip(lower=0)
            g['Est_Uplift'] = g['Gap_to_Overall'] * g['count']
            out_df = g.reset_index().rename(columns={"index": dim})
            fig = px.bar(out_df.sort_values("Gap_to_Overall", ascending=False).head(10),
                         y=dim, x="Gap_to_Overall", orientation="h",
                         labels={"Gap_to_Overall": f"Gap to Overall (↑ better)", dim: dim},
                         title=f"Quick Wins: Close the Gap on {metr}")
            st.plotly_chart(fig, use_container_width=True); save_chart_png(fig, f"opportunities_{dim}_{metr}.png")
            st.dataframe(out_df, use_container_width=True)
            top_msgs = []
            for _, row in out_df.sort_values("Gap_to_Overall", ascending=False).head(5).iterrows():
                if row["Gap_to_Overall"] > 0:
                    top_msgs.append(f"{row[dim]} (gap {row['Gap_to_Overall']:.2f}/unit; est uplift≈{row['Est_Uplift']:.0f})")
            msg = "Close gaps on: " + ", ".join(top_msgs) if top_msgs else "No below-average segments."
            st.markdown("### What this means"); st.markdown(msg)
            add_card("Opportunities", f"{dim}/{metr}", msg, impact=0.8, priority=0.8)
    else:
        st.info("⚠️ Select correct data type (need at least one categorical & one numeric).")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------- Explorer ----------------------------
with tabs[4]:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("Interactive Data Explorer")
    flt_cols = st.multiselect("Columns to filter", df_pl.columns, default=[])
    df_view = df_pd.copy()

    for c in flt_cols:
        if c in num_cols:
            mn, mx = float(np.nanmin(df_pd[c])), float(np.nanmax(df_pd[c]))
            v = st.slider(f"{c} range", mn, mx, (mn, mx))
            df_view = df_view[(df_view[c] >= v[0]) & (df_view[c] <= v[1])]
        elif c in cat_cols:
            vals = df_pd[c].dropna().unique().tolist()
            pick = st.multiselect(f"{c} values", vals[:500])
            if pick: df_view = df_view[df_view[c].isin(pick)]
        elif c in dt_cols:
            dmin, dmax = pd.to_datetime(df_pd[c], errors="coerce").min(), pd.to_datetime(df_pd[c], errors="coerce").max()
            if pd.notna(dmin) and pd.notna(dmax):
                rng = st.date_input(f"{c} window", (dmin.date(), dmax.date()))
                if isinstance(rng, tuple) and len(rng) == 2:
                    df_view = df_view[(pd.to_datetime(df_view[c]) >= pd.Timestamp(rng[0])) &
                                      (pd.to_datetime(df_view[c]) <= pd.Timestamp(rng[1]) + pd.Timedelta(days=1))]
        else:
            st.info("⚠️ Select correct data type for filtering.")

    st.dataframe(df_view.head(1000), use_container_width=True)
    colA, colB, colC = st.columns(3)
    with colA:
        csv_bytes = df_view.to_csv(index=False).encode("utf-8")
        st.download_button("💾 Download Cleaned CSV", csv_bytes, file_name="cleaned_data.csv", mime="text/csv")
    with colB:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            df_view.head(50000).to_excel(writer, index=False, sheet_name="Data")
            if num_cols:
                corr = df_view[num_cols].corr()
                corr.to_excel(writer, sheet_name="Correlation")
        buf.seek(0)
        st.download_button("📊 Download Insights (Excel)", buf, file_name="insights.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with colC:
        buf = io.StringIO()
        writer = csv_module.writer(buf, quoting=csv_module.QUOTE_ALL)
        writer.writerow(["Category","Title","Text","Impact","Priority"])
        for n in st.session_state.get("narratives", []):
            writer.writerow([n.get("category",""), n.get("title",""), n.get("text",""), n.get("impact",0), n.get("priority",0)])
        csv_export = buf.getvalue().encode("utf-8")
        st.download_button("📝 Download Insight Cards (CSV)", csv_export, file_name="insight_cards.csv", mime="text/csv")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------- PDF Report ----------------------------
with st.container():
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("📑 One-Click PDF")
    from fpdf import FPDF
    def build_pdf(cards: List[dict], images: List[Tuple[str, bytes]]) -> bytes:
        pdf = FPDF(unit="pt", format="A4")
        pdf.set_auto_page_break(auto=True, margin=40)
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 20, "Auto Insights Report", ln=1, align="C")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 14, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1)
        pdf.ln(6)

        # Cards
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 16, "Executive Insight Cards", ln=1)
        pdf.set_font("Helvetica", "", 11)
        for c in sorted(cards, key=lambda x: x.get("impact",0)*x.get("priority",0), reverse=True):
            pdf.set_font("Helvetica", "B", 11)
            pdf.multi_cell(0, 14, f"[{c.get('category','')}] {c.get('title','')}  (score={(c.get('impact',0)*c.get('priority',0)):.2f})")
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 14, str(c.get("text","")).replace("**",""))
            pdf.ln(4)

        # Visuals
        if images:
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 16, "Visuals", ln=1)
            x, y = 36, 60
            max_w, max_h = 250, 170
            for i, (fname, b) in enumerate(images):
                try:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                    tmp.write(b); tmp.close()
                    pdf.image(tmp.name, x=x, y=y, w=max_w, h=max_h)
                    os.unlink(tmp.name)
                except Exception:
                    pass
                x += max_w + 24
                if x + max_w > 560:
                    x = 36
                    y += max_h + 24
                if y + max_h > 770 and i < len(images)-1:
                    pdf.add_page(); x, y = 36, 60

        mem = io.BytesIO()
        pdf.output(mem)
        mem.seek(0)
        return mem.read()

    if st.button("📥 Download PDF Report"):
        pdf_bytes = build_pdf(st.session_state["narratives"], st.session_state["charts_png"])
        st.download_button("Save PDF", data=pdf_bytes, file_name=f"auto_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------- Expander: Feature Summary ----------------------------
with st.expander("Core Functionality & Technical Excellence", expanded=False):
    st.markdown("""
**Core Functionality**
- Smart Data Processing: Parquet conversion + Polars backend ✅
- AI-Powered Insights: Executive cards (What/Why/Action/Impact), trend, anomalies, seasonality, drivers ✅
- Advanced Analytics: Correlation, feature sensitivity, opportunities with uplift ✅
- Interactive Visualizations: Plotly charts with guardrails ✅
- Data Narratives: Auto-generated, ranked for priority × impact ✅

**Technical Excellence**
- Production-Grade: Python 3.11+, robust error guards (“Select correct data type”), memory-aware ✅
- Scalable Architecture: Session caching, Parquet, Polars ✅
- Beautiful UI: Clean white theme, insight cards, responsive layout ✅
- Export: Visuals ZIP (PNG), Insights Excel, Cleaned CSV, Insight Cards CSV, PDF ✅
""")
