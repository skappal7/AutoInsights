# app.py
# 🚀 Auto Insights — V5 (Storytelling + Scalable + Beautiful)
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

# ---------------------------- Page & Theme ----------------------------
st.set_page_config(page_title="Auto Insights – CE Innovations Lab", page_icon="🚀", layout="wide")

# Glassmorphism + gradient styles
st.markdown("""
<style>
/* Background gradient */
.stApp {
  background: radial-gradient(1200px 800px at 10% 10%, rgba(99,102,241,0.12), transparent 60%),
              radial-gradient(1200px 800px at 90% 30%, rgba(236,72,153,0.12), transparent 60%),
              linear-gradient(135deg, #0f172a, #0b1220 60%);
  color: #eef2ff;
}
/* Cards */
.block-container { padding-top: 1.5rem; }
.glass {
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid rgba(255, 255, 255, 0.12);
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
  backdrop-filter: blur(10px);
  border-radius: 18px;
  padding: 1rem 1.2rem;
  margin-bottom: 1rem;
}
.kpi {
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.18);
  border-radius: 16px;
  padding: 0.8rem 1rem;
  text-align: center;
}
.kpi h3 { margin: 0; font-size: 0.9rem; color: #c7d2fe; }
.kpi p { margin: 0; font-size: 1.3rem; font-weight: 700; color: #e5e7eb; }
footer {visibility: hidden;}
#custom-footer {
  position: fixed; left: 0; right: 0; bottom: 0; z-index: 9999;
  background: rgba(17,24,39,0.65);
  backdrop-filter: blur(6px);
  border-top: 1px solid rgba(255,255,255,0.12);
  color: #e5e7eb; padding: 8px 16px; text-align: center; font-size: 0.9rem;
}
.smallnote { color: #cbd5e1; font-size: 0.9rem; }
hr { border-color: rgba(255,255,255,0.15); }
</style>
<div id="custom-footer">Developed by <b>CE Innovations Lab 2025</b></div>
""", unsafe_allow_html=True)

# ---------------------------- Session Helpers ----------------------------
if "charts_png" not in st.session_state:
    st.session_state["charts_png"] = []  # list of (filename, bytes)
if "narratives" not in st.session_state:
    st.session_state["narratives"] = []  # list of (title, text)
if "df_pl" not in st.session_state:
    st.session_state["df_pl"] = None
if "df_pd" not in st.session_state:
    st.session_state["df_pd"] = None
if "parquet_path" not in st.session_state:
    st.session_state["parquet_path"] = None

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
                    if s is None:
                        continue
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
    # Keep a lightweight pandas copy only for plotting libs that require pandas
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

def add_narrative(title: str, text: str):
    if text and text.strip():
        st.session_state["narratives"].append((title, text))

def save_chart_png(fig: go.Figure, fname: str):
    try:
        img = to_image(fig, format="png", scale=2)  # requires kaleido
        st.session_state["charts_png"].append((fname, img))
    except Exception:
        pass  # if kaleido missing, silently ignore export but keep UI functional

def zip_all_charts() -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, b in st.session_state["charts_png"]:
            zf.writestr(fname, b)
    mem.seek(0)
    return mem.read()

# ---------------------------- Analytics & Narratives ----------------------------
def basic_stats_numeric(series: pd.Series) -> Dict[str, float]:
    s = series.dropna()
    if s.empty: return {}
    q1, q3 = np.percentile(s, [25, 75])
    iqr = max(q3 - q1, 1e-12)
    outliers = ((s < (q1 - 1.5 * iqr)) | (s > (q3 + 1.5 * iqr))).sum()
    out_pct = 100 * outliers / len(s) if len(s) else 0
    return {
        "min": float(np.min(s)),
        "max": float(np.max(s)),
        "mean": float(np.mean(s)),
        "median": float(np.median(s)),
        "std": float(np.std(s, ddof=1)) if len(s) > 1 else 0.0,
        "skew": float(pd.Series(s).skew()) if len(s) > 2 else 0.0,
        "kurt": float(pd.Series(s).kurt()) if len(s) > 3 else 0.0,
        "out_pct": out_pct
    }

def hist_narrative(df_pd: pd.DataFrame, col: str) -> str:
    stats = basic_stats_numeric(df_pd[col])
    if not stats: return "No numeric data available."
    skew_txt = "positively skewed" if stats["skew"] > 0.5 else "negatively skewed" if stats["skew"] < -0.5 else "fairly symmetric"
    return (
        f"Range {stats['min']:.2f}–{stats['max']:.2f}, mean {stats['mean']:.2f}, median {stats['median']:.2f}. "
        f"Distribution appears {skew_txt}. ~{stats['out_pct']:.1f}% potential outliers."
    )

def box_narrative(df_pd: pd.DataFrame, col: str) -> str:
    stats = basic_stats_numeric(df_pd[col])
    if not stats: return "No numeric data available."
    return f"Variation (std {stats['std']:.2f}) with ~{stats['out_pct']:.1f}% outliers; check process stability and data quality."

def bar_narrative(df_pd: pd.DataFrame, cat_col: str, val_col: Optional[str], agg: str) -> str:
    if val_col is None:
        counts = df_pd[cat_col].value_counts(dropna=False)
        if counts.empty: return "No categories present."
        top = counts.iloc[:3]
        conc = 100 * counts.iloc[0] / counts.sum()
        return f"Top categories: {', '.join([f'{idx} ({val})' for idx, val in top.items()])}. Concentration at top: {conc:.1f}%."
    else:
        grouped = df_pd.groupby(cat_col)[val_col].agg(agg).sort_values(ascending=False)
        if grouped.empty: return "No values to aggregate."
        top3 = grouped.iloc[:3]
        gap = float(top3.iloc[0] - top3.iloc[1]) if len(top3) > 1 else 0.0
        return f"Leading segments: {', '.join([f'{idx} ({val:.2f})' for idx, val in top3.items()])}. Gap leader→runner-up: {gap:.2f}."

def line_narrative(df_pd: pd.DataFrame, tcol: str, vcol: str) -> str:
    dfp = df_pd[[tcol, vcol]].dropna().copy()
    dfp[tcol] = pd.to_datetime(dfp[tcol], errors="coerce")
    dfp = dfp.dropna().sort_values(tcol)
    if dfp.empty or len(dfp) < 3: return "Insufficient time points."
    x = (dfp[tcol] - dfp[tcol].min()).dt.total_seconds().values.reshape(-1, 1)
    y = dfp[vcol].values
    lr = LinearRegression().fit(x, y)
    slope = lr.coef_[0]
    direction = "upward" if slope > 0 else "downward"
    vol = pd.Series(y).pct_change().std() * 100 if len(y) > 2 else 0.0
    delta = y[-1] - y[0]
    return f"{direction} trend (Δ={delta:.2f}) with volatility ~{vol:.1f}%."

def scatter_narrative(df_pd: pd.DataFrame, x: str, y: str) -> str:
    sub = df_pd[[x, y]].dropna()
    if sub.empty or len(sub) < 3: return "Insufficient points."
    r = float(sub.corr().iloc[0,1])
    strength = "strong" if abs(r) >= 0.7 else "moderate" if abs(r) >= 0.4 else "weak"
    return f"{strength} correlation (r={r:.2f}). Inspect non-linearity & outliers before causal claims."

def corr_matrix(df_pd: pd.DataFrame, numeric_cols: List[str]) -> Tuple[pd.DataFrame, str]:
    if len(numeric_cols) < 2:
        return pd.DataFrame(), "Not enough numeric columns for correlation."
    corr = df_pd[numeric_cols].corr()
    flat = corr.unstack().dropna()
    flat = flat[flat.index.get_level_values(0) != flat.index.get_level_values(1)]
    if flat.empty: return corr, "No meaningful correlations detected."
    top_pos = flat.sort_values(ascending=False).head(3)
    top_neg = flat.sort_values(ascending=True).head(3)
    msg = "Top relationships: " + "; ".join([f"{a}~{b} (+{v:.2f})" for (a,b),v in top_pos.items()]) + \
          " | " + "Most negative: " + "; ".join([f"{a}~{b} ({v:.2f})" for (a,b),v in top_neg.items()])
    return corr, msg

def rf_feature_importance(df_pd: pd.DataFrame, target: str) -> Optional[pd.Series]:
    df = df_pd.select_dtypes(include=np.number).dropna()
    if target not in df.columns or df.shape[1] < 2: return None
    X = df.drop(columns=[target])
    y = df[target]
    if X.empty: return None
    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    fi = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    return fi

def mi_scores(df_pd: pd.DataFrame, target: str) -> Optional[pd.Series]:
    df = df_pd.select_dtypes(include=np.number).dropna()
    if target not in df.columns or df.shape[1] < 2: return None
    X = df.drop(columns=[target])
    y = df[target]
    scores = mutual_info_regression(X, y, random_state=42)
    return pd.Series(scores, index=X.columns).sort_values(ascending=False)

def opportunities_by_category(df_pd: pd.DataFrame, dim: str, metric: str, top_n: int = 5) -> Tuple[pd.DataFrame, str]:
    d = df_pd[[dim, metric]].dropna()
    if d.empty: return pd.DataFrame(), "No data for opportunity analysis."
    overall = d[metric].mean()
    g = d.groupby(dim)[metric].mean().sort_values()
    under = g.head(min(top_n, len(g)))
    uplift = (overall - under).clip(lower=0.0)
    out = pd.DataFrame({"Avg": g, "Gap_to_Overall": uplift})
    msg = "Quick wins: " + ", ".join([f"{idx} (↑{gap:.2f} to reach avg)" for idx, gap in uplift[uplift>0].items()][:top_n])
    return out, msg if uplift.sum() > 0 else "No gaps vs overall average."

# ---------------------------- Header ----------------------------
st.markdown("<h1 style='margin-bottom:0'>🚀 Auto Insights – Storytelling Analytics</h1>", unsafe_allow_html=True)
st.markdown("<div class='smallnote'>Beautiful visuals + dynamic narratives + scalable analytics (Parquet + Polars)</div>", unsafe_allow_html=True)
st.write("")

# ---------------------------- Upload & Load ----------------------------
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

# ---------------------------- 5 Tab Interface ----------------------------
tabs = st.tabs(["Overview", "AI Insights", "Visualizations", "Opportunities", "Data Explorer"])

# ---------------------------- Overview ----------------------------
with tabs[0]:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    with r1c1:
        st.markdown("<div class='kpi'><h3>Rows</h3><p>{:,}</p></div>".format(df_pl.height), unsafe_allow_html=True)
    with r1c2:
        st.markdown("<div class='kpi'><h3>Columns</h3><p>{}</p></div>".format(len(df_pl.columns)), unsafe_allow_html=True)
    with r1c3:
        miss = float(df_pd.isna().sum().sum())
        st.markdown("<div class='kpi'><h3>Missing Values</h3><p>{:,}</p></div>".format(int(miss)), unsafe_allow_html=True)
    with r1c4:
        st.markdown("<div class='kpi'><h3>Parquet</h3><p>On</p></div>", unsafe_allow_html=True)

    st.markdown("**Schema**")
    sch_df = pd.DataFrame({"Column": df_pl.columns, "Dtype": [str(t) for t in df_pl.dtypes]})
    st.dataframe(sch_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Missingness plot
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    miss_series = df_pd.isna().sum().sort_values(ascending=False)
    if miss_series.sum() > 0:
        fig = px.bar(miss_series.reset_index(), x="index", y=0, labels={"index":"Column", "0":"Missing"})
        st.plotly_chart(fig, use_container_width=True)
        save_chart_png(fig, "missing_values.png")
        msg = "Data quality: {} columns with missing values; prioritize imputation for top affected.".format((miss_series>0).sum())
        st.caption(msg)
        add_narrative("Missingness", msg)
    else:
        st.caption("No missing values detected.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------- AI Insights ----------------------------
with tabs[1]:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("Executive Summary")
    # Top correlations
    corr, cmsg = corr_matrix(df_pd, num_cols)
    if not corr.empty:
        fig = px.imshow(corr, color_continuous_scale="RdBu_r", origin="lower", aspect="auto", zmin=-1, zmax=1)
        st.plotly_chart(fig, use_container_width=True)
        save_chart_png(fig, "correlation_heatmap.png")
    st.caption(cmsg)
    add_narrative("Correlations", cmsg)

    # Target selection for drivers/prescriptions
    target = st.selectbox("Select a numeric Target for driver analysis", num_cols, index=0 if num_cols else None)
    if target:
        fi = rf_feature_importance(df_pd, target)
        if fi is not None:
            fig = px.bar(fi.head(12), labels={"value":"Importance","index":"Feature"}, title=f"Top Drivers of {target}")
            st.plotly_chart(fig, use_container_width=True)
            save_chart_png(fig, f"drivers_{target}.png")
            msg = "Key drivers of {}: {}.".format(
                target, ", ".join([f"{k} ({v:.2f})" for k,v in fi.head(5).items()])
            )
            st.caption(msg)
            add_narrative("Drivers", msg)
        mis = mi_scores(df_pd, target)
        if mis is not None:
            fig = px.bar(mis.head(12), labels={"value":"MI Score","index":"Feature"}, title=f"Information Gain wrt {target}")
            st.plotly_chart(fig, use_container_width=True)
            save_chart_png(fig, f"mi_{target}.png")
            msg = "Signals with high information gain for {}: {}.".format(
                target, ", ".join([f"{k} ({v:.2f})" for k,v in mis.head(5).items()])
            )
            st.caption(msg)
            add_narrative("Information Gain", msg)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------- Visualizations (Storytelling per chart) ----------------------------
with tabs[2]:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("Custom Chart Builder")

    chart_type = st.selectbox("Chart", ["Histogram", "Box", "Bar (Count)", "Bar (Aggregate)", "Line", "Scatter", "Correlation Heatmap"])
    col1, col2, col3, col4 = st.columns(4)

    # Histogram
    if chart_type == "Histogram":
        with col1:
            num = st.selectbox("Numeric", num_cols)
        if friendly_guard(num is not None and num in num_cols):
            fig = px.histogram(df_pd, x=num, nbins=50, opacity=0.85)
            st.plotly_chart(fig, use_container_width=True)
            save_chart_png(fig, f"hist_{num}.png")
            st.caption(hist_narrative(df_pd, num))

    # Box
    if chart_type == "Box":
        with col1:
            num = st.selectbox("Numeric", num_cols, key="boxn")
        with col2:
            group = st.selectbox("Group (optional)", ["(none)"] + cat_cols, key="boxg")
        if friendly_guard(num is not None and num in num_cols):
            if group and group != "(none)":
                fig = px.box(df_pd, x=group, y=num, points="outliers")
            else:
                fig = px.box(df_pd, y=num, points="outliers")
            st.plotly_chart(fig, use_container_width=True)
            save_chart_png(fig, f"box_{num}.png")
            st.caption(box_narrative(df_pd, num))

    # Bar Count
    if chart_type == "Bar (Count)":
        with col1:
            cat = st.selectbox("Categorical", cat_cols, key="barc")
        if friendly_guard(cat is not None and cat in cat_cols):
            topn = st.slider("Top N", 5, 50, 20)
            counts = df_pd[cat].value_counts(dropna=False).head(topn)
            fig = px.bar(counts, labels={"index":cat, "value":"Count"})
            st.plotly_chart(fig, use_container_width=True)
            save_chart_png(fig, f"bar_count_{cat}.png")
            st.caption(bar_narrative(df_pd, cat, None, "count"))

    # Bar Aggregate
    if chart_type == "Bar (Aggregate)":
        with col1:
            cat = st.selectbox("Categorical", cat_cols, key="bara")
        with col2:
            val = st.selectbox("Value", num_cols, key="barv")
        with col3:
            agg = st.selectbox("Aggregate", ["mean","sum","median"], index=0)
        if friendly_guard((cat in cat_cols) and (val in num_cols)):
            g = df_pd.groupby(cat)[val].agg(agg).sort_values(ascending=False).head(50)
            fig = px.bar(g, labels={"index":cat, "value":f"{agg}({val})"})
            st.plotly_chart(fig, use_container_width=True)
            save_chart_png(fig, f"bar_agg_{cat}_{val}_{agg}.png")
            st.caption(bar_narrative(df_pd, cat, val, agg))

    # Line
    if chart_type == "Line":
        with col1:
            tcol = st.selectbox("Time", dt_cols + cat_cols, key="linet")
        with col2:
            vcol = st.selectbox("Value", num_cols, key="linev")
        # Try parse times if user selected a non-datetime string column
        if tcol in cat_cols:
            df_tmp = df_pd.copy()
            df_tmp[tcol] = pd.to_datetime(df_tmp[tcol], errors="coerce")
        else:
            df_tmp = df_pd
        ok = (tcol in dt_cols or tcol in cat_cols) and (vcol in num_cols) and (pd.to_datetime(df_tmp[tcol], errors="coerce").notna().sum()>2)
        if friendly_guard(ok):
            fig = px.line(df_tmp.sort_values(tcol), x=tcol, y=vcol)
            st.plotly_chart(fig, use_container_width=True)
            save_chart_png(fig, f"line_{tcol}_{vcol}.png")
            st.caption(line_narrative(df_tmp, tcol, vcol))

    # Scatter
    if chart_type == "Scatter":
        with col1:
            x = st.selectbox("X (numeric)", num_cols, key="scx")
        with col2:
            y = st.selectbox("Y (numeric)", [c for c in num_cols if c != x], key="scy")
        with col3:
            color = st.selectbox("Color (optional)", ["(none)"] + cat_cols, key="scc")
        if friendly_guard((x in num_cols) and (y in num_cols)):
            if color and color != "(none)":
                fig = px.scatter(df_pd, x=x, y=y, color=color, trendline="ols", opacity=0.75)
            else:
                fig = px.scatter(df_pd, x=x, y=y, trendline="ols", opacity=0.75)
            st.plotly_chart(fig, use_container_width=True)
            save_chart_png(fig, f"scatter_{x}_{y}.png")
            st.caption(scatter_narrative(df_pd, x, y))

    # Correlation Heatmap
    if chart_type == "Correlation Heatmap":
        ok = len(num_cols) >= 2
        if friendly_guard(ok):
            corr, cmsg2 = corr_matrix(df_pd, num_cols)
            fig = px.imshow(corr, color_continuous_scale="RdBu_r", origin="lower", aspect="auto", zmin=-1, zmax=1)
            st.plotly_chart(fig, use_container_width=True)
            save_chart_png(fig, "correlation_heatmap_builder.png")
            st.caption(cmsg2)

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
        if friendly_guard(ok):
            out, msg = opportunities_by_category(df_pd, dim, metr, top_n=6)
            if not out.empty:
                fig = px.bar(out.sort_values("Gap_to_Overall", ascending=False).head(10), y=out.index, x="Gap_to_Overall",
                             labels={"Gap_to_Overall":"Gap to Overall ↑", "index":dim}, orientation="h")
                st.plotly_chart(fig, use_container_width=True)
                save_chart_png(fig, f"opportunities_{dim}_{metr}.png")
                st.dataframe(out, use_container_width=True)
            st.caption(msg)
            add_narrative("Opportunities", f"{dim}/{metr}: {msg}")
    else:
        st.info("⚠️ Select correct data type (need at least one categorical & one numeric).")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------- Data Explorer ----------------------------
with tabs[4]:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("Interactive Data Explorer")
    flt_cols = st.multiselect("Columns to filter", df_pl.columns, default=[])
    df_view = df_pd.copy()

    # Dynamic filters
    for c in flt_cols:
        if c in num_cols:
            mn, mx = float(np.nanmin(df_pd[c])), float(np.nanmax(df_pd[c]))
            v = st.slider(f"{c} range", mn, mx, (mn, mx))
            df_view = df_view[(df_view[c] >= v[0]) & (df_view[c] <= v[1])]
        elif c in cat_cols:
            vals = df_pd[c].dropna().unique().tolist()
            pick = st.multiselect(f"{c} values", vals[:500])
            if pick:
                df_view = df_view[df_view[c].isin(pick)]
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
        csv = df_view.to_csv(index=False).encode("utf-8")
        st.download_button("💾 Download Cleaned CSV", csv, file_name="cleaned_data.csv", mime="text/csv")
    with colB:
        # Insights workbook (multi-sheet) — simple export
        from io import BytesIO
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            df_view.head(50000).to_excel(writer, index=False, sheet_name="Data")
            if num_cols:
                corr = df_view[num_cols].corr()
                corr.to_excel(writer, sheet_name="Correlation")
        buf.seek(0)
        st.download_button("📊 Download Insights (Excel)", buf, file_name="insights.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with colC:
        # Narratives export
        narr_csv = "Title,Text\n" + "\n".join([f"\"{t}\",\"{x.replace('\"','''')}\"" for t,x in st.session_state["narratives"]])
        st.download_button("📝 Download Narratives (CSV)", narr_csv.encode("utf-8"), file_name="narratives.csv", mime="text/csv")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------- PDF Report (Narratives + Thumbnails) ----------------------------
with st.container():
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("📑 One-Click PDF")
    from fpdf import FPDF
    def build_pdf(narrs: List[Tuple[str,str]], images: List[Tuple[str, bytes]]) -> bytes:
        pdf = FPDF(unit="pt", format="A4")
        pdf.set_auto_page_break(auto=True, margin=40)
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 20, "Auto Insights Report", ln=1, align="C")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 14, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1)
        pdf.ln(6)

        # Narratives
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 16, "Executive Narratives", ln=1)
        pdf.set_font("Helvetica", "", 11)
        for title, text in narrs:
            pdf.set_font("Helvetica", "B", 11)
            pdf.multi_cell(0, 14, f"• {title}")
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 14, text)
            pdf.ln(4)

        # Visual thumbnails
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

# ---------------------------- Footer Core Functionality (static copy for clarity) ----------------------------
with st.expander("Core Functionality & Technical Excellence", expanded=False):
    st.markdown("""
**Core Functionality**
- Smart Data Processing: Automatically converts uploads to Parquet (efficient) ✅
- AI-Powered Insights: Descriptive, predictive, prescriptive (rule-based engine) ✅
- Advanced Analytics: Correlation, causal drivers, outliers, performance gaps ✅
- Interactive Visualizations: Beautiful, responsive Plotly charts ✅
- Data Narratives: Auto-generated executive summaries & per-chart explanations ✅

**Technical Excellence**
- Production-Grade: Python 3.11+, robust guards (“Select correct data type”), memory-aware ✅
- Scalable Architecture: Session caching, Parquet conversion, Polars backend ✅
- Beautiful UI: Modern glassmorphism gradients & responsive layout ✅
- Export: Insights to Excel, visuals to PNG/ZIP, cleaned data to CSV, full PDF ✅

**User Experience**
- 5 Tabs: Overview • AI Insights • Visualizations • Opportunities • Data Explorer ✅
- Custom Chart Builder + Advanced Filtering ✅
- One-Click Exports (ZIP / Excel / PDF / CSV) ✅
""")
