import streamlit as st
import pandas as pd
from src.manager import Manager
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt

# Set up simple streamlit frontend for python LLM query of irae data
st.set_page_config(
    page_title = "irAE Dataset LLM Assistant",
    layout = "wide"
)

# Set a single global default (optional)
pio.templates.default = "seaborn"

# Set default color scheme
pio.templates["seaborn"].layout.colorway = [
    "rgba(70,120,150,0.8)",  
    "rgba(110,140,170,0.8)", 
    "rgba(150,170,190,0.8)", 
    "rgba(200,140,120,0.8)", 
    "rgba(120,150,110,0.8)", 
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
            tr.update(
                marker=dict(
                    size=6,
                    opacity=0.85,
                    color="rgba(70,120,150,0.8)",
                    line=dict(width=0.5, color="black")
                ),
                line=dict(width=2, color="rgba(70,120,150,1)")
            )

        # Bar / box / violin
        elif t in ["bar", "box", "violin"]:
            if hasattr(tr, "marker"):
                if tr.marker.color is None:
                    tr.marker.color = None
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
@st.cache_data
def load_data():
    messy_data = pd.read_csv("data/data_david_new.csv", sep = "$")

    # Replace any empty strings with NaN
    messy_data.replace("", pd.NA, inplace=True)

    # Change all "_" to "," so that rows with multiple entries are comma-separated always
    string_cols = messy_data.select_dtypes(include='object').columns
    for col in string_cols:
        messy_data[col] = messy_data[col].str.replace("_", ",", regex=False)
        messy_data[col] = messy_data[col].str.title() # Also capitalize first letter of each word

    # Standardize any columns containing comma-separated values
    for col in messy_data.columns:

    # Identify whether any columns contain comma-separated entries suggesting multiple values per row
        if messy_data[col].astype(str).str.contains(",").any():
            messy_data[col] = messy_data[col].str.replace(r"\s*,\s*", ",", regex=True) # Ensure no spaces after commas
            messy_data[col] = messy_data[col].str.strip() # Remove leading/trailing spaces
            messy_data[col] = messy_data[col].str.replace(r"^,\s*|\s*,\s*$", "", regex=True) # Remove leading/trailing commas 

    # Make a year column 
    messy_data['year'] = messy_data['quarter'].str.slice(0, 4)

    # Get rid of columns where the comma-separated values are merged into "other"
    cols_to_drop = ['irae_type','brand_name','tumor_type','ici_drug_name','drug_class']
    for col in cols_to_drop:
        if col in messy_data.columns:
            messy_data.drop(columns=col, inplace=True)

    rename_dict = {
        'irae_type_expanded': 'irae_type',
        'ici_drug_name_expanded': 'ici_drug_name',
        'brand_name_expanded': 'brand_name',
        'drug_class_expanded': 'drug_class',
        'tumor_type_expanded': 'tumor_type'
    }

    messy_data.rename(columns=rename_dict, inplace=True)

    return messy_data 

# Clean and load data 
df = load_data()

# Instantiate manager class
m = Manager(df)

# Setup app layout 
st.title("irAE Dataset LLM Assistant")
st.markdown(
    "Ask natural language questions about the FAERS irAE dataset."
    " The assistant will generate python code to compute or visualize the result."
)

# Input question box
question = st.text_input("Ask a question:", placeholder="e.g. Show me lung cancer patients with a rash.")
run_btn = st.button("Submit", width="stretch")

# Main logic
if run_btn and question:
    with st.spinner("Analyzing your question..."):
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
        elif res_type == "plot" and isinstance(res_data, plt.Figure) or hasattr(res_data, "figure"):
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








