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


# TRY also including option to upload data
# Just testing this, delete later
# But proof of concept that my setup could be used for any table
# st.sidebar.header("Data Options")

# uploaded_file = st.sidebar.file_uploader(
#     "Upload your own dataset (optional)", type=["csv"]
# )

# if uploaded_file is not None:
#     st.sidebar.success("Custom dataset uploaded successfully!")
#     df=pd.read_csv(uploaded_file)
#     df.replace("", pd.NA, inplace=True)
    
# else:
#     st.sidebar.info("Using default FAERS irAE dataset.")
#     df = load_data()

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




