# This function will read in irae data from David, clean/process it, and return a new csv file
# This will be the source of the irae data used for the LLM query

import pandas as pd
from pathlib import Path

#############
### Paths ###
#############
PATH_TO_TABLE = Path("~/Documents/Tan_Lab/Projects/Python_irAE_LLM_Query/data/irae_data_raw.csv").expanduser()
OUTPUT_PATH = PATH_TO_TABLE.parent / "irae_data_cleaned.csv"

#######################################
##### Main data cleaning function #####
#######################################
def clean_irae_data(input_path):

    # Read in the data
    df = pd.read_csv(input_path, sep="$")

    # Convert col names to lowercase
    df.columns = [col.lower() for col in df.columns]

    # Replace empty strings with NaN
    df.replace("", pd.NA, inplace=True)

    # Convert date columns to correct dtype
    for col in df.columns:
        if "date" in col:
            df[col] = df[col].replace(0, pd.NA)
            df[col] = pd.to_datetime(df[col], format="%Y%m%d", errors="coerce")

    # Change all "_" to "," so that rows with multiple entries are comma-separated always
    string_cols = df.select_dtypes(include='object').columns
    for col in string_cols:
        df[col] = df[col].str.replace("_", ",", regex=False)
        df[col] = df[col].str.title() # Also capitalize first letter of each word

    # Clean up some specific problem columns
    # 1. 'cancertype' column 
    if 'cancertype' in df.columns:
        df['cancertype'] = df['cancertype'].str.replace(r"(?i)\s*Cancer", "", regex=True) # Remove 'cancer' (case insensitive) and any leading spaces
        df['cancertype'] = df['cancertype'].str.replace(r"\s*,\s*", ", ", regex=True) # Ensure single space after commas
        df['cancertype'] = df['cancertype'].str.strip() # Remove leading/trailing spaces
        df['cancertype'] = df['cancertype'].str.replace(r"^,\s*|\s*,\s*$", "", regex=True) # Remove leading/trailing commas

    # 2. 'irae_category_expanded' column
    if 'irae_category_expanded' in df.columns:
        df['irae_category_expanded'] = df['irae_category_expanded'].str.replace(r"(?i)\s*Toxicities", "", regex=True) # Remove 'toxicities' (case insensitive) and any leading spaces
        df['irae_category_expanded'] = df['irae_category_expanded'].str.replace(r"\s*,\s*", ", ", regex=True) # Ensure single space after commas
        df['irae_category_expanded'] = df['irae_category_expanded'].str.strip() # Remove leading/trailing spaces
        df['irae_category_expanded'] = df['irae_category_expanded'].str.replace(r"^,\s*|\s*,\s*$", "", regex=True) # Remove leading/trailing commas 

    # 3. Recode age_grp column
    if 'age_grp' in df.columns:
        df["age_grp"] = df["age_grp"].replace({
            "E": "Elderly",
            "T": "Teen",
            "A": "Adult",
            "C": "Child",
            "I": "Infant",
            "N": "Neonate"
        })

    # 4. Recode combination column
    if "combination" in df.columns:
        df["combination"] = df["combination"].replace({
            "Io - Combo": "Combo Immunotherapy",
            "Io - Single": "Single",
            "Io + Chemo": "Chemotherapy/Immunotherapy",
            "Io + Targeted Therapy": "Targeted Therapy/Immunotherapy"
        })


    # Drop unnecessary columns if they exist
    cols_to_drop = ["prod_ai", "drug_class", "iraecategory", "frame", "role_cod", "drug_seq"]
    for col in cols_to_drop:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    # Rename columns to have more distinct and descriptive names for LLM 
    rename_dict = {
        "primaryid": "patient_id",
        "drugname": "drug_name",
        "outc_cod": "outcome",
        "drug_class_expanded": "drug_class",
        "irae_category_expanded": "irae_type",
        "cancertype": "tumor_type",
        "age_grp": "age_group"
    }

    df.rename(columns=rename_dict, inplace=True)

    return df


#####################################
# Run the cleaning and save as a csv
#####################################
if __name__ == "__main__":
    cleaned_df = clean_irae_data(PATH_TO_TABLE)
    cleaned_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Cleaned data saved to {OUTPUT_PATH}")





