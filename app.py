import streamlit as st
import polars as pl
import altair as alt
import numpy as np
import plotly.express as px
from io import BytesIO
from fpdf import FPDF

st.set_page_config(page_title="🚀 Auto Insights V4", layout="wide")

# ================== Load Data ==================
@st.cache_data(show_spinner=False)
def load_data(file) -> pl.DataFrame:
    ext = file.name.split(".")[-1].lower()
    if ext == "csv":
        df = pl.read_csv(file)
    elif ext in ["xlsx", "xls"]:
        import pandas as pd
        pdf = pd.read_excel(file)
        df = pl.from_pandas(pdf)
    else:
        raise ValueError("Unsupported format")
    return df

# ================== Narrative Generators ==================
def describe_distribution(df: pl.DataFrame, col: str) -> str:
    series = df[col]
    mean, std, minv, maxv = series.mean(), series.std(), series.min(), series.max()
    skewness = ((series - mean)**3).mean() / (std**3 + 1e-9)
    trend = "positively skewed" if skewness > 0.5 else "negatively skewed" if skewness < -0.5 else "fairly symmetric"
    return f"📊 **{col}** ranges {minv:.2f} → {maxv:.2f} (mean={mean:.2f}, std={std:.2f}). Distribution is {trend}."

def bar_narrative(df: pl.DataFrame, col: str) -> str:
    counts = df[col].value_counts().to_pandas()
    top_cat, top_val = counts.iloc[0][col], counts.iloc[0]['count']
    narr = f"📊 **{col}** distribution: Top category is **{top_cat}** with {top_val} occurrences. "
    if len(counts) > 1:
        second_val = counts.iloc[1]['count']
        narr += f"Gap between top and second is {top_val - second_val}."
    return narr

def line_narrative(df: pl.DataFrame, time_col: str, val_col: str) -> str:
    series = df.select([time_col, val_col]).to_pandas().sort_values(time_col)[val_col]
    growth = series.iloc[-1] - series.iloc[0]
    trend = "increasing" if growth > 0 else "decreasing"
    return f"📈 Over time, **{val_col}** is {trend} by {growth:.2f} units from start to end."

def scatter_narrative(df: pl.DataFrame, x: str, y: str) -> str:
    corr = df.select([x, y]).to_pandas().corr().iloc[0, 1]
    strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.4 else "weak"
    return f"🔗 Scatter shows {strength} correlation ({corr:.2f}) between **{x}** and **{y}**."

def correlation_narrative(df: pl.DataFrame):
    num_cols = df.select(pl.col(pl.Float64), pl.col(pl.Int64)).columns
    if len(num_cols) < 2:
        return "Not enough numeric columns for correlation."
    corr = df.select(num_cols).to_pandas().corr()
    top_corr = corr.unstack().dropna().sort_values(ascending=False)
    top_corr = top_corr[top_corr < 0.999].head(3)
    narr = "📌 Top correlations:\n"
    for (c1, c2), val in top_corr.items():
        narr += f"- {c1} vs {c2} → {val:.2f}\n"
    return narr

# ================== PDF Report ==================
def generate_pdf(narratives):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "Auto Insights Report", ln=True, align="C")
    pdf.ln(10)
    for title, text in narratives.items():
        pdf.set_font("Arial", "B", 12)
        pdf.cell(200, 10, title, ln=True)
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 8, text)
        pdf.ln(5)
    buf = BytesIO()
    pdf.output(buf)
    buf.seek(0)
    st.download_button("📑 Download Full Report", buf, "auto_insights_report.pdf", mime="application/pdf")

# ================== Streamlit App ==================
st.title("🚀 Auto Insights V4 – Full Data Storytelling")
st.markdown("Every visual comes with **dynamic narratives** powered by Polars + Altair.")

uploaded_file = st.file_uploader("Upload dataset", type=["csv","xlsx","xls"])
if uploaded_file:
    df = load_data(uploaded_file)
    st.success("✅ Data loaded")
    narratives = {}

    # Overview
    st.subheader("📌 Data Overview")
    st.write(df.head(10).to_pandas())
    overview = f"Dataset has {df.shape[0]:,} rows and {df.shape[1]} columns."
    st.markdown(overview)
    narratives["Overview"] = overview

    # Distribution
    st.subheader("📊 Distribution")
    col = st.selectbox("Choose numeric column", df.columns)
    if col:
        chart = alt.Chart(df.to_pandas()).mark_bar().encode(x=alt.X(col, bin=alt.Bin(maxbins=30)), y='count()')
        st.altair_chart(chart, use_container_width=True)
        narr = describe_distribution(df, col)
        st.markdown(narr)
        narratives["Distribution"] = narr

    # Bar Chart
    st.subheader("📊 Bar Chart (Categorical)")
    cat_col = st.selectbox("Choose categorical column", df.columns)
    if cat_col:
        chart = alt.Chart(df.to_pandas()).mark_bar().encode(x=cat_col, y='count()')
        st.altair_chart(chart, use_container_width=True)
        narr = bar_narrative(df, cat_col)
        st.markdown(narr)
        narratives["Bar Chart"] = narr

    # Line Chart
    st.subheader("📈 Line Chart (Trend)")
    time_col = st.selectbox("Choose time column", df.columns)
    val_col = st.selectbox("Choose value column", df.columns)
    if time_col and val_col:
        chart = alt.Chart(df.to_pandas()).mark_line().encode(x=time_col, y=val_col)
        st.altair_chart(chart, use_container_width=True)
        narr = line_narrative(df, time_col, val_col)
        st.markdown(narr)
        narratives["Line Chart"] = narr

    # Scatter Plot
    st.subheader("🔗 Scatter Plot")
    x_col = st.selectbox("X-axis", df.columns)
    y_col = st.selectbox("Y-axis", df.columns, index=min(1, len(df.columns)-1))
    if x_col and y_col:
        chart = alt.Chart(df.to_pandas()).mark_circle(size=60, opacity=0.7).encode(x=x_col, y=y_col, tooltip=list(df.columns))
        st.altair_chart(chart, use_container_width=True)
        narr = scatter_narrative(df, x_col, y_col)
        st.markdown(narr)
        narratives["Scatter Plot"] = narr

    # Correlation Heatmap
    st.subheader("📈 Correlation Heatmap")
    num_cols = df.select(pl.col(pl.Float64), pl.col(pl.Int64)).columns
    if len(num_cols) > 1:
        corr = df.select(num_cols).to_pandas().corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
        st.plotly_chart(fig, use_container_width=True)
        narr = correlation_narrative(df)
        st.markdown(narr)
        narratives["Correlation Heatmap"] = narr

    # Report
    st.subheader("📑 Export Report")
    generate_pdf(narratives)
