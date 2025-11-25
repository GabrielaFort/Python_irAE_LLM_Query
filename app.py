import streamlit as st
import pandas as pd
from src.manager import Manager
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
import numpy as np
from src.utils import load_data
from matplotlib_venn._common import VennDiagram


# Set up simple streamlit frontend for python LLM query of irae data
st.set_page_config(
    page_title = "irAE Dataset LLM Assistant",
    layout = "wide",
    initial_sidebar_state = "collapsed"
)

# Some custom styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
    }
    .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3, h4 {
        color: #22303C;
    }
    .dataframe {
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Set a single global default (optional)
pio.templates.default = "seaborn"

# Set default color scheme
pio.templates["seaborn"].layout.colorway = [
    "rgba(70,120,150,0.8)",  
    "rgba(110,140,170,0.8)", 
    "rgba(150,170,190,0.8)", 
    "rgba(200,140,120,0.8)", 
    "rgba(120,150,110,0.8)",
    "rgba(180,180,100,0.8)",
    "rgba(160,110,160,0.8)",
    "rgba(100,160,180,0.8)",
    "rgba(210,160,100,0.8)",
    "rgba(90,130,100,0.8)"
]

# Helper function for plotly styling
def apply_default_style(fig):
    """Apply a single, consistent style safely across Plotly trace types."""
    # ---- Layout (titles, axes, legend, margins)
    fig.update_layout(
        template="seaborn",
        font=dict(size=14, color="#222222"),
        title_font=dict(size=16, color="#111111"),
        legend=dict(
            title_font=dict(size=14),
            font=dict(size=13),
            bgcolor="rgba(0,0,0,0)"
        ),
        xaxis=dict(
            title_font=dict(size=14, color="#111111"),
            tickfont=dict(size=12, color="#333333"),
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)",
            zeroline=False,
            linecolor="rgba(0,0,0,0.5)"
        ),
        yaxis=dict(
            title_font=dict(size=14, color="#111111"),
            tickfont=dict(size=12, color="#333333"),
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)",
            zeroline=False,
            linecolor="rgba(0,0,0,0.5)"
        ),
        margin=dict(l=60, r=40, t=60, b=60)
    )

    # ---- Per-trace styling ----
    for tr in fig.data:
        t = (tr.type or "").lower()

        # Scatter / line
        if t in ["scatter", "scattergl", "line"]:
            has_multicolor = (hasattr(tr, "marker") and isinstance(getattr(tr.marker, "color", None), (list, np.ndarray)))
            tr.update(
                marker=dict(
                    size=6,
                    opacity=0.85,
                    color=None if has_multicolor else getattr(tr.marker, "color", "rgba(70,120,150,0.8)"),
                    line=dict(width=0.5, color="black")
                ),
                line=dict(width=2, color="rgba(70,120,150,1)")
            )

        # Bar / box / violin
        elif t in ["bar", "box", "violin"]:
            if hasattr(tr, "marker"):
                if getattr(tr.marker, "color", None) is None:
                    tr.marker.color = "rgba(70,120,150,0.8)"
                tr.marker.line = dict(width=1, color="black")

        # Heatmap / surface
        elif t in ["heatmap", "surface", "contour"]:
            tr.update(colorscale="Viridis", showscale=True)

        # Pie / donut
        elif t in ["pie", "donut"]:
            tr.update(textfont=dict(size=13, color="#222222"))

    return fig


# Matplot lib venn viagram style
plt.rcParams.update({"font.size": 14})


# Load dataset from David
#@st.cache_data

# Clean and load data 
df = load_data()

# Instantiate manager class
m = Manager(df)

# Setup app layout 
st.title("irAE Dataset LLM Assistant")
st.markdown("""
Welcome to the **irAE Dataset LLM Assistant**, a natural-language interface to explore immune-related adverse events (irAEs) reported in the **FAERS** dataset.

Use this tool to:
- Ask questions about specific cancer types, drugs, or toxicities  
- Generate plots or summaries of irAE patterns  
- Automatically produce reproducible Python code

---
""")

# Data Overview Section
st.header("Dataset Overview")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Number of Records", f"{df.shape[0]:,}")
with col2:
    st.metric("Number of Columns", f"{df.shape[1]}")

st.caption("Below is a preview of the FAERS irAE dataset:")
st.dataframe(df.head(10), width='stretch', hide_index=True)

# Download section
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Full Dataset (CSV)",
    data=csv,
    file_name="irae_faers_dataset.csv",
    mime="text/csv",
    help="Download the entire dataset as a CSV file."
)

with st.expander("View column descriptions"):
    st.markdown("""
    - **patient_id**: Unique identifier for each patient  
    - **irae**: irAE(s) reported
    - **irae_type**: Broader category of irAE
    - **outcome**: Patient outcome 
    - **ici_drug_name**: Immunotherapy drug(s) administered 
    - **brand_name**: Brand name(s) of immunotherapy drug(s) administered  
    - **drug_class**: Class of immunotherapy drug(s)
    - **cancer_drug_name**: Other anti-cancer or chemotherapy drugs administered
    - **combination status**: Whether immunotherapy was given in combination with other drugs
    - **other_drug_name**: Other non-cancer and non-ici drugs administered
    - **tumor_type**: Reported primary cancer
    - **time_to_onset**: Weeks from drug start to irAE onset
    - **age**: Patient age in years
    - **age_group**: Groups by age range
    - **sex**: Patient sex
    - **quarter**: FAERS reporting quarter
    - **year**: FAERS reporting year   
    """)

st.markdown("---")


### Query interface ###
st.header("Ask a Question")
# Input question box with Enter submission
with st.form("query_form", clear_on_submit=False):
    question = st.text_input("Ask a question:", placeholder="e.g. Show me lung cancer patients with a rash.")
    submitted = st.form_submit_button("Submit", width='stretch')

if submitted and question:
    with st.spinner("Processing your question..."):
        result = m.process_question(question)

    res_type = result.get('type')
    res_data = result.get('data')
    res_code = result.get('code')

    # Tabs for organizing results
    tab_result, tab_code = st.tabs(["Result","Generated Code"])

    with tab_result:

        # For plotly plots
        if res_type == "plotly" and isinstance(res_data, go.Figure):

            plotly_config = {"displayModeBar": True,
                            "scrollZoom": True,
                            "responsive": True,
                            "editable": True}
            fig = apply_default_style(res_data)
            st.plotly_chart(fig, config=plotly_config)

        # Handle matplotlib plots like venn diagrams
        elif res_type == "plot" and isinstance(res_data, plt.Figure) or hasattr(res_data, "figure") or isinstance(res_data, VennDiagram):
            st.pyplot(res_data, clear_figure=True)

        elif res_type == "dataframe":
            st.dataframe(res_data, width="stretch", hide_index=True)
            st.write(f"Result has {res_data.shape[0]} rows and {res_data.shape[1]} columns.")

        elif res_type == "number":
            st.metric(label="Result", value = res_data)

        elif res_type == "error":
            st.error(res_data)
        
        else:
            st.write(res_data)

    with tab_code:
        st.code(res_code or "No code generated.", language = "python")

else:
    st.info("Enter a question above to get started.")








