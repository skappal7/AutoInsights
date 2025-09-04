import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.inspection import permutation_importance
import warnings
import io
import base64
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq
import json
from typing import Dict, List, Tuple, Any
import plotly.io as pio
from pathlib import Path
import tempfile
import zipfile

warnings.filterwarnings('ignore')

# Configure Streamlit
st.set_page_config(
    page_title="AutoInsights Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .insight-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    .narrative-section {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    
    .opportunity-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

class DataProcessor:
    """Advanced data processing and analysis engine"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def convert_to_parquet(self, df: pd.DataFrame) -> bytes:
        """Convert DataFrame to parquet format for efficient storage"""
        table = pa.Table.from_pandas(df)
        parquet_buffer = io.BytesIO()
        pq.write_table(table, parquet_buffer)
        return parquet_buffer.getvalue()
    
    def load_from_parquet(self, parquet_data: bytes) -> pd.DataFrame:
        """Load DataFrame from parquet bytes"""
        return pd.read_parquet(io.BytesIO(parquet_data))
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced data cleaning and preprocessing"""
        df_clean = df.copy()
        
        # Handle missing values intelligently
        for col in df_clean.columns:
            if df_clean[col].dtype in ['int64', 'float64']:
                # Fill numeric columns with median
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            else:
                # Fill categorical columns with mode
                mode_val = df_clean[col].mode()
                if not mode_val.empty:
                    df_clean[col].fillna(mode_val.iloc[0], inplace=True)
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        
        # Convert date columns
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                # Try to convert to datetime
                try:
                    df_clean[col] = pd.to_datetime(df_clean[col], errors='ignore')
                except:
                    pass
        
        return df_clean
    
    def detect_data_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Intelligently detect and categorize column types"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Remove datetime columns that were originally categorical
        categorical_cols = [col for col in categorical_cols if col not in datetime_cols]
        
        return {
            'numeric': numeric_cols,
            'categorical': categorical_cols,
            'datetime': datetime_cols
        }

class InsightGenerator:
    """Advanced insight generation engine"""
    
    def __init__(self, df: pd.DataFrame, data_types: Dict[str, List[str]]):
        self.df = df
        self.data_types = data_types
        self.insights = []
        self.narratives = []
        self.opportunities = []
        
    def generate_descriptive_insights(self) -> List[Dict[str, Any]]:
        """Generate comprehensive descriptive insights"""
        insights = []
        
        # Dataset overview
        insights.append({
            'type': 'overview',
            'title': 'Dataset Overview',
            'content': f"Dataset contains {len(self.df)} rows and {len(self.df.columns)} columns. "
                      f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
            'severity': 'info'
        })
        
        # Missing data analysis
        missing_data = self.df.isnull().sum()
        if missing_data.sum() > 0:
            worst_col = missing_data.idxmax()
            insights.append({
                'type': 'data_quality',
                'title': 'Data Quality Alert',
                'content': f"Found missing values in {missing_data[missing_data > 0].count()} columns. "
                          f"'{worst_col}' has the most missing values ({missing_data[worst_col]:,} missing, "
                          f"{missing_data[worst_col]/len(self.df)*100:.1f}%)",
                'severity': 'warning'
            })
        
        # Numeric insights
        for col in self.data_types['numeric']:
            if self.df[col].std() > 0:  # Avoid constant columns
                skewness = self.df[col].skew()
                
                if abs(skewness) > 1:
                    skew_type = "highly skewed" if abs(skewness) > 2 else "moderately skewed"
                    direction = "right" if skewness > 0 else "left"
                    
                    insights.append({
                        'type': 'distribution',
                        'title': f'Distribution Analysis: {col}',
                        'content': f"'{col}' is {skew_type} to the {direction} (skewness: {skewness:.2f}). "
                                  f"Median: {self.df[col].median():.2f}, Mean: {self.df[col].mean():.2f}",
                        'severity': 'info'
                    })
                
                # Outlier detection
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((self.df[col] < (Q1 - 1.5 * IQR)) | 
                           (self.df[col] > (Q3 + 1.5 * IQR))).sum()
                
                if outliers > 0:
                    insights.append({
                        'type': 'outliers',
                        'title': f'Outlier Detection: {col}',
                        'content': f"Found {outliers} outliers in '{col}' ({outliers/len(self.df)*100:.1f}% of data). "
                                  f"These may represent data entry errors or genuine extreme values requiring investigation.",
                        'severity': 'warning'
                    })
        
        # Categorical insights
        for col in self.data_types['categorical']:
            if self.df[col].nunique() < len(self.df) * 0.8:  # Avoid high cardinality
                value_counts = self.df[col].value_counts()
                most_common = value_counts.iloc[0]
                
                insights.append({
                    'type': 'categorical',
                    'title': f'Category Analysis: {col}',
                    'content': f"'{col}' has {self.df[col].nunique()} unique values. "
                              f"Most common: '{value_counts.index[0]}' ({most_common:,} occurrences, "
                              f"{most_common/len(self.df)*100:.1f}%)",
                    'severity': 'info'
                })
        
        return insights
    
    def find_correlations(self) -> List[Dict[str, Any]]:
        """Advanced correlation analysis"""
        correlations = []
        numeric_cols = self.data_types['numeric']
        
        if len(numeric_cols) >= 2:
            corr_matrix = self.df[numeric_cols].corr()
            
            # Find strong correlations
            for i, col1 in enumerate(numeric_cols):
                for j, col2 in enumerate(numeric_cols[i+1:], i+1):
                    corr_val = corr_matrix.loc[col1, col2]
                    
                    if abs(corr_val) > 0.5:  # Strong correlation threshold
                        strength = "very strong" if abs(corr_val) > 0.8 else "strong"
                        direction = "positive" if corr_val > 0 else "negative"
                        
                        correlations.append({
                            'type': 'correlation',
                            'title': f'Strong Correlation Found',
                            'content': f"'{col1}' and '{col2}' show {strength} {direction} correlation "
                                      f"(r = {corr_val:.3f}). This suggests these variables move together.",
                            'severity': 'info',
                            'variables': [col1, col2],
                            'correlation': corr_val
                        })
        
        return correlations
    
    def perform_causal_inference(self) -> List[Dict[str, Any]]:
        """Advanced causal inference and feature importance analysis"""
        causal_insights = []
        numeric_cols = self.data_types['numeric']
        
        if len(numeric_cols) >= 2:
            # Try to identify potential target variables (highly correlated with others)
            corr_matrix = self.df[numeric_cols].corr()
            avg_correlations = corr_matrix.abs().mean().sort_values(ascending=False)
            
            potential_targets = avg_correlations.head(3).index.tolist()
            
            for target in potential_targets:
                if len(numeric_cols) > 1:
                    features = [col for col in numeric_cols if col != target]
                    
                    if len(features) > 0:
                        try:
                            X = self.df[features].fillna(self.df[features].median())
                            y = self.df[target].fillna(self.df[target].median())
                            
                            # Random Forest for feature importance
                            rf = RandomForestRegressor(n_estimators=100, random_state=42)
                            rf.fit(X, y)
                            
                            feature_importance = pd.DataFrame({
                                'feature': features,
                                'importance': rf.feature_importances_
                            }).sort_values('importance', ascending=False)
                            
                            top_driver = feature_importance.iloc[0]
                            
                            causal_insights.append({
                                'type': 'causal',
                                'title': f'Key Driver Identified: {target}',
                                'content': f"'{top_driver['feature']}' is the strongest predictor of '{target}' "
                                          f"(importance: {top_driver['importance']:.3f}). "
                                          f"Focus on this variable for maximum impact on {target}.",
                                'severity': 'success',
                                'target': target,
                                'driver': top_driver['feature'],
                                'importance': top_driver['importance']
                            })
                            
                        except Exception as e:
                            continue
        
        return causal_insights
    
    def identify_opportunities(self) -> List[Dict[str, Any]]:
        """Identify actionable business opportunities"""
        opportunities = []
        
        # Performance gaps analysis
        numeric_cols = self.data_types['numeric']
        
        for col in numeric_cols:
            if self.df[col].std() > 0:
                # Identify potential improvement opportunities
                q75 = self.df[col].quantile(0.75)
                q25 = self.df[col].quantile(0.25)
                
                low_performers = (self.df[col] <= q25).sum()
                high_performers = (self.df[col] >= q75).sum()
                
                if low_performers > 0:
                    potential_improvement = q75 - q25
                    
                    opportunities.append({
                        'type': 'improvement',
                        'title': f'Performance Gap: {col}',
                        'content': f"{low_performers:,} records in '{col}' are in the bottom 25th percentile. "
                                  f"Bringing them to top quartile level could improve values by up to "
                                  f"{potential_improvement:.2f} units on average.",
                        'severity': 'opportunity',
                        'impact': 'high' if low_performers > len(self.df) * 0.2 else 'medium',
                        'affected_records': low_performers,
                        'improvement_potential': potential_improvement
                    })
        
        # Category concentration opportunities
        for col in self.data_types['categorical']:
            if self.df[col].nunique() < 20:  # Manageable number of categories
                value_counts = self.df[col].value_counts()
                
                if len(value_counts) > 1:
                    top_category_pct = value_counts.iloc[0] / len(self.df)
                    
                    if top_category_pct < 0.5:  # Fragmented distribution
                        opportunities.append({
                            'type': 'consolidation',
                            'title': f'Category Consolidation: {col}',
                            'content': f"'{col}' shows fragmented distribution with top category representing only "
                                      f"{top_category_pct*100:.1f}% of data. Consider consolidating similar categories "
                                      f"or investigating why distribution is so dispersed.",
                            'severity': 'opportunity',
                            'impact': 'medium'
                        })
        
        return opportunities
    
    def generate_narrative(self) -> str:
        """Generate comprehensive data narrative"""
        narrative_parts = []
        
        # Executive Summary
        narrative_parts.append(f"""
        ## 📊 Executive Summary
        
        Your dataset contains **{len(self.df):,}** records across **{len(self.df.columns)}** variables, providing a rich foundation for analysis. 
        Here's what the data reveals:
        """)
        
        # Data composition
        numeric_count = len(self.data_types['numeric'])
        categorical_count = len(self.data_types['categorical'])
        datetime_count = len(self.data_types['datetime'])
        
        narrative_parts.append(f"""
        ### 🏗️ Data Structure
        - **{numeric_count}** numerical variables for quantitative analysis
        - **{categorical_count}** categorical variables for segmentation
        - **{datetime_count}** time-based variables for temporal analysis
        """)
        
        # Key insights narrative
        if self.data_types['numeric']:
            main_numeric = self.data_types['numeric'][0]
            mean_val = self.df[main_numeric].mean()
            std_val = self.df[main_numeric].std()
            
            narrative_parts.append(f"""
            ### 📈 Key Patterns
            Looking at '{main_numeric}' as a primary metric, we observe an average of **{mean_val:.2f}** 
            with a standard deviation of **{std_val:.2f}**, indicating {'high variability' if std_val/mean_val > 0.3 else 'moderate consistency'} 
            in the data.
            """)
        
        # Correlation insights
        numeric_cols = self.data_types['numeric']
        if len(numeric_cols) >= 2:
            corr_matrix = self.df[numeric_cols].corr()
            max_corr = corr_matrix.abs().unstack().sort_values(ascending=False)
            max_corr = max_corr[max_corr < 1.0]  # Remove self-correlations
            
            if len(max_corr) > 0:
                top_corr = max_corr.iloc[0]
                vars_names = max_corr.index[0]
                
                narrative_parts.append(f"""
                ### 🔗 Relationships
                The strongest relationship exists between **{vars_names[0]}** and **{vars_names[1]}** 
                (correlation: {top_corr:.3f}), suggesting these variables are closely linked and should be 
                considered together in decision-making.
                """)
        
        # Opportunities summary
        narrative_parts.append(f"""
        ### 🎯 Strategic Opportunities
        Based on the analysis, several optimization opportunities emerge:
        
        1. **Data Quality Enhancement**: Address missing values and outliers for improved accuracy
        2. **Performance Optimization**: Focus on underperforming segments identified in the analysis  
        3. **Predictive Modeling**: Leverage strong correlations for forecasting and planning
        """)
        
        return "\n".join(narrative_parts)

class VisualizationEngine:
    """Advanced visualization generation"""
    
    def __init__(self, df: pd.DataFrame, data_types: Dict[str, List[str]]):
        self.df = df
        self.data_types = data_types
        
    def create_overview_dashboard(self) -> List[go.Figure]:
        """Create comprehensive overview visualizations"""
        figures = []
        
        # Dataset overview metrics
        fig = go.Figure()
        
        metrics = [
            ['Rows', len(self.df)],
            ['Columns', len(self.df.columns)],
            ['Numeric Vars', len(self.data_types['numeric'])],
            ['Categorical Vars', len(self.data_types['categorical'])],
            ['Missing Values', self.df.isnull().sum().sum()]
        ]
        
        fig.add_trace(go.Bar(
            x=[m[0] for m in metrics],
            y=[m[1] for m in metrics],
            marker_color=['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe']
        ))
        
        fig.update_layout(
            title="Dataset Overview Metrics",
            template="plotly_dark",
            height=400
        )
        
        figures.append(fig)
        
        # Correlation heatmap for numeric variables
        if len(self.data_types['numeric']) >= 2:
            numeric_df = self.df[self.data_types['numeric']].select_dtypes(include=[np.number])
            corr_matrix = numeric_df.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0
            ))
            
            fig.update_layout(
                title="Correlation Matrix - Numeric Variables",
                template="plotly_dark",
                height=500
            )
            
            figures.append(fig)
        
        return figures
    
    def create_distribution_plots(self) -> List[go.Figure]:
        """Create advanced distribution visualizations"""
        figures = []
        
        # Numeric distributions
        for col in self.data_types['numeric'][:4]:  # Limit to first 4 for performance
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=[f'{col} - Histogram', f'{col} - Box Plot']
            )
            
            # Histogram
            fig.add_trace(
                go.Histogram(x=self.df[col], nbinsx=30, name='Distribution'),
                row=1, col=1
            )
            
            # Box plot
            fig.add_trace(
                go.Box(y=self.df[col], name='Box Plot'),
                row=1, col=2
            )
            
            fig.update_layout(
                title=f"Distribution Analysis: {col}",
                template="plotly_dark",
                height=400,
                showlegend=False
            )
            
            figures.append(fig)
        
        return figures
    
    def create_insights_visualizations(self, insights: List[Dict[str, Any]]) -> List[go.Figure]:
        """Create visualizations for specific insights"""
        figures = []
        
        # Feature importance visualization from causal insights
        causal_insights = [i for i in insights if i.get('type') == 'causal']
        
        if causal_insights:
            for insight in causal_insights[:2]:  # Limit to 2 for performance
                target = insight.get('target')
                driver = insight.get('driver')
                
                if target and driver:
                    # Scatter plot showing relationship
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=self.df[driver],
                        y=self.df[target],
                        mode='markers',
                        marker=dict(
                            color=self.df[target],
                            colorscale='viridis',
                            size=6,
                            opacity=0.7
                        ),
                        name=f'{driver} vs {target}'
                    ))
                    
                    # Add trend line
                    z = np.polyfit(self.df[driver].fillna(0), self.df[target].fillna(0), 1)
                    p = np.poly1d(z)
                    fig.add_trace(go.Scatter(
                        x=sorted(self.df[driver]),
                        y=p(sorted(self.df[driver])),
                        mode='lines',
                        name='Trend Line',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig.update_layout(
                        title=f"Key Relationship: {driver} drives {target}",
                        xaxis_title=driver,
                        yaxis_title=target,
                        template="plotly_dark",
                        height=400
                    )
                    
                    figures.append(fig)
        
        return figures

class ExportManager:
    """Handle all export functionality"""
    
    @staticmethod
    def create_download_link(data: bytes, filename: str, mime_type: str) -> str:
        """Create download link for data"""
        b64 = base64.b64encode(data).decode()
        return f'<a href="data:{mime_type};base64,{b64}" download="{filename}">📥 Download {filename}</a>'
    
    @staticmethod
    def export_insights_to_excel(insights: List[Dict[str, Any]], opportunities: List[Dict[str, Any]]) -> bytes:
        """Export insights and opportunities to Excel"""
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Insights sheet
            insights_df = pd.DataFrame([
                {
                    'Type': i.get('type', ''),
                    'Title': i.get('title', ''),
                    'Content': i.get('content', ''),
                    'Severity': i.get('severity', '')
                } for i in insights
            ])
            insights_df.to_excel(writer, sheet_name='Insights', index=False)
            
            # Opportunities sheet
            opportunities_df = pd.DataFrame([
                {
                    'Type': o.get('type', ''),
                    'Title': o.get('title', ''),
                    'Content': o.get('content', ''),
                    'Impact': o.get('impact', ''),
                    'Affected Records': o.get('affected_records', 0)
                } for o in opportunities
            ])
            opportunities_df.to_excel(writer, sheet_name='Opportunities', index=False)
        
        return output.getvalue()
    
    @staticmethod
    def export_visualizations(figures: List[go.Figure]) -> bytes:
        """Export all visualizations as images in a zip file"""
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for i, fig in enumerate(figures):
                img_bytes = pio.to_image(fig, format='png', width=1200, height=800)
                zip_file.writestr(f'visualization_{i+1}.png', img_bytes)
        
        return zip_buffer.getvalue()

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'insights' not in st.session_state:
    st.session_state.insights = []
if 'opportunities' not in st.session_state:
    st.session_state.opportunities = []
if 'narrative' not in st.session_state:
    st.session_state.narrative = ""
if 'visualizations' not in st.session_state:
    st.session_state.visualizations = []

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🚀 AutoInsights Analytics Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Transform your data into actionable insights with AI-powered analytics</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Choose your data file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload CSV or Excel files for analysis"
        )
        
        if uploaded_file is not None:
            # Cache the uploaded file processing
            if st.session_state.processed_data is None or st.button("🔄 Refresh Analysis"):
                with st.spinner("🔄 Processing your data..."):
                    try:
                        # Load data
                        if uploaded_file.name.endswith('.csv'):
                            df = pd.read_csv(uploaded_file)
                        else:
                            df = pd.read_excel(uploaded_file)
                        
                        # Process data
                        processor = DataProcessor()
                        df_clean = processor.clean_data(df)
                        data_types = processor.detect_data_types(df_clean)
                        
                        # Generate insights
                        insight_gen = InsightGenerator(df_clean, data_types)
                        insights = insight_gen.generate_descriptive_insights()
                        correlations = insight_gen.find_correlations()
                        causal = insight_gen.perform_causal_inference()
                        opportunities = insight_gen.identify_opportunities()
                        narrative = insight_gen.generate_narrative()
                        
                        # Generate visualizations
                        viz_engine = VisualizationEngine(df_clean, data_types)
                        overview_viz = viz_engine.create_overview_dashboard()
                        dist_viz = viz_engine.create_distribution_plots()
                        insight_viz = viz_engine.create_insights_visualizations(causal)
                        
                        # Store in session state
                        st.session_state.processed_data = df_clean
                        st.session_state.data_types = data_types
                        st.session_state.insights = insights + correlations + causal
                        st.session_state.opportunities = opportunities
                        st.session_state.narrative = narrative
                        st.session_state.visualizations = overview_viz + dist_viz + insight_viz
                        
                        st.success("✅ Analysis completed successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Error processing file: {str(e)}")
                        return
        
# Export visualizations
if st.button("🖼️ Export Visualizations"):
    try:
        viz_zip = ExportManager.export_visualizations(st.session_state.visualizations)
        st.download_button(
            label="📥 Download Visualizations",
            data=viz_zip,
            file_name=f"visualizations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip",
        )
    except Exception as e:
        # Plotly image export needs 'kaleido' installed; this makes failures clearer.
        st.error(
            "❌ Failed to export images. "
            "If the error mentions 'kaleido', install it with: pip install -U kaleido. "
            f"Details: {e}"
        )
