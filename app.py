import streamlit as st
import pandas as pd
from src.manager import Manager
from src.agents.explanation_agent import ExplanationAgent
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
import numpy as np
from src.utils import build_context, explanation_llm 
from matplotlib_venn._common import VennDiagram
from src.agents.guideline_agent import link_short_citations
import urllib.parse
import re

# Set up simple streamlit frontend for python LLM query of irae data
st.set_page_config(
    page_title = "irAE Dataset LLM Assistant",
    layout = "wide",
    initial_sidebar_state = "collapsed"
)

# Initialize session state
if "history" not in st.session_state:
    st.session_state["history"] = []

if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

if "last_explanation" not in st.session_state:
    st.session_state["last_explanation"] = None

if "pending_question" not in st.session_state:
    st.session_state["pending_question"] = None

if "rerun_query" not in st.session_state:
    st.session_state["rerun_query"] = None

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
                line=dict(width=2)
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

# load cleaned data
@st.cache_data
def load_data():
    return pd.read_csv("data/irae_data_cleaned.csv")
df = load_data()

# Load FAISS index ONCE globally (shared across all sessions)
@st.cache_resource
def load_index_manager():
    """Load the FAISS index manager once and share across sessions."""
    from src.index_manager import IndexManager
    index_mgr = IndexManager(
        kb_dir="src/knowledge_base", 
        model_name="NeuML/pubmedbert-base-embeddings"
    )
    index_mgr.load()
    return index_mgr

shared_index = load_index_manager()

# Instantiate session-specific manager
if "manager" not in st.session_state:
    st.session_state["manager"] = Manager(df.copy(deep=True), shared_index_manager=shared_index)
m = st.session_state["manager"]

# Instantiate session-specific explanation agent
if "explanation_agent" not in st.session_state:
    st.session_state["explanation_agent"] = ExplanationAgent(explanation_llm())
explanation_agent = st.session_state["explanation_agent"]

# Setup app layout 
st.title("irAE Dataset LLM Assistant")

# Introduction section
st.markdown("""
Welcome to the **irAE Dataset LLM Assistant**, a natural-language interface to explore immune-related adverse events (irAEs) reported in the **FAERS** dataset.

Use this tool to:
- Ask questions about specific cancer types, drugs, toxicities, or current irAE guidelines  
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
#csv = df.to_csv(index=False).encode('utf-8')
#st.download_button(
#    label="Download Full Dataset (CSV)",
#    data=csv,
#    file_name="irae_faers_dataset.csv",
#    mime="text/csv",
#    help="Download the entire dataset as a CSV file."
#)

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

# Pre-load question into text box if rerun is scheduled
if st.session_state["pending_question"] is not None:
    st.session_state["question_input"] = st.session_state["pending_question"]
    st.session_state["pending_question"] = None

### Query interface ###
st.header("Ask a Question")

# Input question box with Enter submission
with st.form("query_form", clear_on_submit=False, enter_to_submit=True):
    question = st.text_input("Ask a question:", placeholder="e.g. Show me lung cancer patients with a rash.",key="question_input", autocomplete="off")
    submitted = st.form_submit_button("Submit", width='stretch')

# Reset session button
if st.button("Reset Conversation"):
    st.session_state["history"] = []
    st.session_state["last_result"] = None
    st.session_state["pending_question"] = None 
    st.session_state["last_explanation"] = None
    st.rerun()

# Handle rerun logic
if st.session_state["rerun_query"]:
    question = st.session_state["rerun_query"]
    st.session_state["rerun_query"] = None

    # Rebuild context
    context = build_context(st.session_state["history"], max_turns=10)

    with st.spinner("Reprocessing your question..."):
        result = m.process_question(question, context=context)
    
    # Add to history immediately 
    code_str = result.get("code") if isinstance(result, dict) else None
    st.session_state["history"].append({"question": question, "code": code_str})

    explanation = None
    if code_str and result.get("type") != "error" and result.get("type") != "text":
        # Rebuild context including latest run
        updated_context = build_context(st.session_state["history"], max_turns=10)
        explanation = explanation_agent.generate_explanation(updated_context)
        st.session_state["last_explanation"] = explanation
    
    else:
        st.session_state["last_explanation"] = None

    st.session_state["last_result"] = result
    st.rerun()  

# Process new question submission 
result = None

if submitted and question:

    # Build LLM memory context from session history
    context = build_context(st.session_state["history"], max_turns=10)

    with st.spinner("Processing your question..."):
        result = m.process_question(question, context=context)

    # Extract LLM generated code for memory
    code_str = result.get("code") if isinstance(result, dict) else None
    
    # Update session history
    st.session_state["history"].append({
            "question": question,
            "code": code_str
        })
    
    # Generate explanation if code was generated
    if code_str and result.get("type") != "error" and result.get("type") != "text":
        # Rebuild context including the latest turn
        updated_context = build_context(st.session_state["history"], max_turns=10)

        explanation = explanation_agent.generate_explanation(updated_context)
        st.session_state["last_explanation"] = explanation

    else:
        st.session_state["last_explanation"] = None

# If no new result but session has history, keep showing the last full result
if result is None:
    result = st.session_state.get("last_result", None)

# Save the last real result separately
if result is not None:
    st.session_state["last_result"] = result

# Display Results
if result is not None:
    res_type = result.get('type')
    res_data = result.get('data')
    res_code = result.get('code')

    # Get explanation from session state
    explanation = st.session_state.get("last_explanation", None)

    code_text = None

    if res_code:
        code_text = str(res_code)

    # Tabs for organizing results
    if code_text:
        tab_result, tab_code = st.tabs(["Result", "Code"])
    else:
        tab_result = st.tabs(["Result"])[0]
        tab_code = None

    with tab_result:

        # Display explanation if available
        if explanation:
            st.info(f"💡 {explanation}")

        # For plotly plots
        if res_type == "plotly" and isinstance(res_data, go.Figure):

            plotly_config = {"displayModeBar": True,
                                "scrollZoom": True,
                                "responsive": True,
                                "editable": True,
                                "toImageButtonOptions": {
                                    "format": "png",
                                    "filename": "irAE_plot",
                                    "scale": 3
                                }}
            
            fig = apply_default_style(res_data)
            st.plotly_chart(fig, config=plotly_config)

        # Handle matplotlib plots like venn diagrams
        elif res_type == "plot" and (isinstance(res_data, plt.Figure)) or hasattr(res_data, "figure") or isinstance(res_data, VennDiagram):
            st.pyplot(res_data, clear_figure=True)

        elif res_type == "dataframe":
            df = res_data.copy()
            for col in df.select_dtypes(include=["float", "float64"]).columns:
                df[col] = df[col].map(
                    lambda x: f"{x:.6f}".rstrip("0").rstrip(".") if pd.notna(x) else x
                    )
            st.dataframe(df,
                width="stretch",
                hide_index=True)
            st.write(f"Result has {res_data.shape[0]} rows and {res_data.shape[1]} columns.")

        elif res_type == "number":
            if isinstance(res_data, (float, np.floating)):
                st.metric(label="Result", value = f"{res_data:.6f}".rstrip("0").rstrip("."))
            else:
                st.metric(label="Result", value = res_data)

        elif res_type == "error":
            st.error(res_data)
            
        else:
            # If the returned data is a string, show it with links for (ASCO), (NCCN), (SITC)
            if isinstance(res_data, str):
                # render RAG output with clickable links if applicable
                html = link_short_citations(res_data)
                st.markdown(html, unsafe_allow_html=True)

            else:
                # non-string results (dict, list, etc.): leave as-is
                st.write(res_data)

    if tab_code:
        with tab_code:
            code_text = str(res_code)
            st.code(code_text, language="python")

else:
    st.info("Enter a question above to get started.")


# Conversation history display
if st.session_state["history"]:
    st.markdown("---")
    st.subheader("Session conversation history")

    for i, turn in enumerate(st.session_state["history"], start=1):
        col_q, col_btn = st.columns([4,1])

        with col_q:
            st.write(f"Q{i}: {turn['question']}")

        with col_btn:
            if st.button("Rerun", key=f"rerun_{i}", help="Rerun this query", width='stretch'):
                st.session_state["pending_question"] = turn['question']
                st.session_state["rerun_query"] = turn['question']
                st.rerun()

