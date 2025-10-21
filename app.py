import streamlit as st
import pandas as pd
from src.manager import Manager

# Set up simple streamlit frontend for python LLM query of irae data
st.set_page_config(
    page_title = "irAE Dataset LLM Assistant",
    layout = "wide"
)

# Load dataset from David
@st.cache_data
def load_data():
    return pd.read_csv("data/data_david_new.csv", sep = "$")

df = load_data()

# Instantiate manager class
m = Manager(df)

# Setup app layout 
st.title("irAE Dataset LLM Assistant")
st.markdown(
    "Ask natural language questions about the FAERS irAE dataset."
    " The assistant will generate python code to compute or visualize the result."
)

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
        if res_type == "plot":
            st.pyplot(result["data"], clear_figure = True)

        elif res_type == "dataframe":
            st.dataframe(res_data, width="stretch", hide_index=True)

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




