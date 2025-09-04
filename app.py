import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from io import BytesIO
from fpdf import FPDF
import os

st.set_page_config(page_title="🚀 Auto Insights V2", layout="wide")

# ================== Utility ==================
@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    ext = os.path.splitext(file.name)[1].lower()
    if ext in [".csv", ".txt"]:
        df = pd.read_csv(file)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file)
    else:
        raise ValueError("Unsupported file format")
    return df

def generate_summary(df: pd.DataFrame) -> str:
    missing = df.isnull().sum().sum()
    num_cols = df.select_dtypes(include=np.number).shape[1]
    cat_cols = df.select_dtypes(exclude=np.number).shape[1]
    return (
        f"Dataset has **{df.shape[0]:,} rows** and **{df.shape[1]} columns**.\n\n"
        f"- Missing values: {missing:,}\n"
        f"- Numeric columns: {num_cols}\n"
        f"- Categorical columns: {cat_cols}\n"
    )

def correlation_analysis(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=np.number).dropna()
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    return corr

def causal_inference(df: pd.DataFrame, target: str):
    df = df.select_dtypes(include=np.number).dropna()
    if target not in df.columns:
        return None
    X, y = df.drop(columns=[target]), df[target]
    if X.empty: return None
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    return fi

def prescriptive_analysis(df: pd.DataFrame, target: str):
    df = df.select_dtypes(include=np.number).dropna()
    if target not in df.columns:
        return None
    X, y = df.drop(columns=[target]), df[target]
    if X.empty: return None
    mi = mutual_info_regression(X, y)
    return pd.Series(mi, index=X.columns).sort_values(ascending=False)

def narrative_from_corr(corr):
    narr = []
    top_corr = corr.unstack().dropna().sort_values(ascending=False)
    top_corr = top_corr[top_corr < 0.999].head(3)  # exclude self
    for (c1, c2), val in top_corr.items():
        narr.append(f"🔗 **{c1}** is highly correlated with **{c2}** (r={val:.2f}).")
    return "\n".join(narr)

def narrative_from_importance(series, title=""):
    if series is None or series.empty: return ""
    top = series.head(3)
    narr = f"📊 Key drivers identified for {title}:\n"
    for feat, score in top.items():
        narr += f"- {feat} ({score:.2f})\n"
    return narr

def download_pdf(narratives, figs):
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
    st.download_button("📑 Download PDF Report", buf, file_name="auto_insights_report.pdf", mime="application/pdf")

# ================== UI ==================
st.title("🚀 Auto Insights V2")
st.markdown("Mind-blowing narratives + visuals + prescriptions.")

uploaded_file = st.file_uploader("Upload your dataset", type=["csv","xlsx","xls","txt"])
if uploaded_file:
    df = load_data(uploaded_file)
    st.success("✅ Data loaded successfully!")
    
    # Sidebar navigation
    menu = st.sidebar.radio("Navigate", ["Overview", "Correlations", "Causal", "Prescriptive", "Opportunities", "Report"])
    
    narratives = {}
    figs = {}

    if menu == "Overview":
        st.subheader("📌 Data Overview")
        st.markdown(generate_summary(df))
        st.dataframe(df.head())

    if menu == "Correlations":
        st.subheader("📈 Correlation Analysis")
        corr = correlation_analysis(df)
        narratives["Correlation Insights"] = narrative_from_corr(corr)

    if menu == "Causal":
        st.subheader("🔍 Causal Inference")
        target = st.selectbox("Select Target Variable", df.select_dtypes(include=np.number).columns)
        fi = causal_inference(df, target)
        if fi is not None:
            st.bar_chart(fi)
            narratives["Causal Inference"] = narrative_from_importance(fi, target)

    if menu == "Prescriptive":
        st.subheader("🧭 Prescriptive Analysis")
        target = st.selectbox("Select Target Variable for Prescriptive", df.select_dtypes(include=np.number).columns, key="presc")
        mi_scores = prescriptive_analysis(df, target)
        if mi_scores is not None:
            st.bar_chart(mi_scores)
            narratives["Prescriptive Insights"] = narrative_from_importance(mi_scores, target)

    if menu == "Opportunities":
        st.subheader("💡 Quick Opportunities")
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) > 0:
            stats = df[num_cols].describe().T
            stats["Opportunity"] = np.where(stats["mean"] < stats["50%"], "Improve low performers", "Maintain high performers")
            st.dataframe(stats[["mean","50%","Opportunity"]])
            narratives["Opportunities"] = "Quick wins identified:\n" + stats["Opportunity"].value_counts().to_string()

    if menu == "Report":
        st.subheader("📑 Auto-Generated Report")
        if narratives:
            for k, v in narratives.items():
                st.markdown(f"### {k}\n{v}")
            download_pdf(narratives, figs)
        else:
            st.info("Generate insights in other tabs first.")
