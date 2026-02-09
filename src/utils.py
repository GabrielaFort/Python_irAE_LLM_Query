# Helpful functions
from streamlit import json
from src.llm_client import LLMClient
import pandas as pd 
import numpy as np
import os
import json
import re
import html
import logging
from pathlib import Path

# Instantiate LLM clients with preset configurations
def question_classifier_llm(): 
    myllm = LLMClient(model="gemini-3-flash-preview:cloud",
                      api_url="https://ollama.com",
                      temperature=0,
                      api_key=os.getenv("OLLAMA_API_KEY"))
    return myllm

def query_llm():
    myllm = LLMClient(model="glm-4.7:cloud",
                api_url="https://ollama.com",
                temperature=0,
                api_key=os.getenv("OLLAMA_API_KEY"))
    return myllm

def plotter_llm():
    myllm = LLMClient(model="gemini-3-flash-preview:cloud",
                api_url="https://ollama.com",
                temperature=0.5,
                api_key=os.getenv("OLLAMA_API_KEY"))
    return myllm

def stats_llm():
    myllm = LLMClient(model="gemini-3-flash-preview:cloud",
                api_url="https://ollama.com",
                temperature=0.1,
                api_key=os.getenv("OLLAMA_API_KEY"))
    return myllm  

def error_checker_llm():
    myllm = LLMClient(model="qwen3-coder:480b-cloud",
                api_url="https://ollama.com",
                temperature=0,
                api_key=os.getenv("OLLAMA_API_KEY"))
    return myllm  

def guideline_llm():
    myllm = LLMClient(model="gpt-oss:120b-cloud",
                api_url="https://ollama.com",
                temperature=0.1,
                api_key=os.getenv("OLLAMA_API_KEY"))
    return myllm

def explanation_llm():
    myllm = LLMClient(model="gemini-3-flash-preview:cloud",
                api_url="https://ollama.com",
                temperature=0.1,
                api_key=os.getenv("OLLAMA_API_KEY"))
    return myllm


# Function to load and clean the irAE dataset
def load_data():
    messy_data = pd.read_csv("data/data_new.csv", sep = "$")

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


# This function will take a dataframe as input and return a summary of the dataframe
# Include data types, column names, and a few example values from each column
# This will be added to the LLM prompts to help the LLM understand the data structure
def summarize_dataframe(df, max_rows=10):
    preview = {}
    hints = []
    dtypes = df.dtypes.astype(str).to_dict()

    # Build a formatted string showing each column and its dtype
    columns_with_types = ", ".join([f"{col} ({dtypes[col]})" for col in df.columns])


    for col in df.columns:
        non_null_values = df[col].dropna().astype(str).unique()[:max_rows]
        preview[col] = list(non_null_values)

    # Identify whether any columns contain comma-separated entries suggesting multiple values per row
        if df[col].astype(str).str.contains(",").any():
            hints.append(f"- Column '{col}' may contain multiple values per record, separated by commas.")

    # Detect ID-like columns
    for col in df.columns:
        nunique = df[col].nunique(dropna = False)
        if nunique == len(df):
            hints.append(f"- Column '{col}' is likely ID column with unique values for each record.")

    # Detect numeric columns and ranges
    for col in df.select_dtypes(include=[np.number]).columns:
        min_val = df[col].min()
        max_val = df[col].max()
        hints.append(f"- Column '{col}' is numeric with range [{min_val}, {max_val}].")

    # Detect categorical columns
    for col in df.select_dtypes(include=['object', 'category']).columns:
        unique_ratio = df[col].nunique(dropna = True) / len(df)
        if unique_ratio < 0.05 and df[col].nunique() < 30: # small number of discrete categories
            hints.append(f"- Column '{col}' is likely categorical with {df[col].nunique()} unique values.")

    # Build hint text
    if hints:
        hint_text = "\n".join(hints)
    else:
        hint_text = "No special structural notes detected."
    
    info = {
        "columns": columns_with_types,
        "num_rows": len(df),
        "num_columns": len(df.columns),
    }

    # Build preview table
    df_preview = pd.DataFrame.from_dict(preview, orient='index').transpose()

    summary = f"""
DataFrame Summary:
Rows: {info['num_rows']}, Columns: {info['num_columns']}
Columns: {info['columns']}
Preview (first {max_rows} unique non-null values per column):
{df_preview.to_markdown(index=False)}

Schema notes:
{hint_text}
"""
    return summary


# Function to clean up code exported from LLM 
# Remove any leading/trailing whitespace and ensure proper indentation
# Remove any markdown formatting if present
def clean_code(code):
    # Remove markdown code block formatting if present
    if code.startswith("```") and code.endswith("```"):
        code = "\n".join(code.split("\n")[1:-1])
    
    # Strip leading/trailing whitespace
    code = code.strip()

    # If code is a single line and doesn't already assign to result, prepend "result = "
    if "\n" not in code and not code.lstrip().startswith("result"):
        code = f"result = {code}"

    # remove import statements if any slipped through
    if "import" in code.lower():
        code = "\n".join([
                line for line in code.split("\n") 
                if not line.strip().startswith(("import ", "from "))
            ])
    return code

# Function to clean text extracted from PDFs or other sources
def clean_text(text):

    # Remove <br> and <br/> tags
    text = re.sub(r"<br\s*/?>", "\n", text)

    # Unescape HTML (e.g., &lt; = <)
    text = html.unescape(text)

    # Replace weird bullet points
    text = text.replace("•", "- ")
    text = text.replace("", "- ")

    # Remove leftover HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    return text.strip()


# Function to build context for prompt using session history gathered through streamlit frontend session state
def build_context(history, max_turns = 10):
    """
    Build conversation history in ollamas message format.
    Args:
        history: list of conversation turns with question and code
        max_turns: maximum number of recent turns to include

    Returns:
        List of message dicts for Ollama API
    """
    messages = []
    
    # Only include the most recent turns
    recent_history = history[-max_turns:] if len(history) > max_turns else history
    
    for turn in recent_history:
        # Add user question
        messages.append({"role": "user", 
                         "content": turn.get("question", "")})
        
        # Add LLM generated code (assistant response)
        if turn.get("code"):
            messages.append({"role": "assistant", 
                             "content": turn.get("code", "")})
                    
    return messages


# Check for malicious code patterns
def is_code_safe(code):
    # Define a list of forbidden patterns
    forbidden_patterns = [
        # --- Imports ---
        r"\bimport\s+os\b",
        r"\bimport\s+sys\b",
        r"\bimport\s+subprocess\b",
        r"\bimport\s+pathlib\b",
        r"\bimport\s+shutil\b",
        r"\bimport\s+pickle\b",
        r"\bfrom\s+os\b",
        r"\bfrom\s+sys\b",
        r"\bfrom\s+subprocess\b",
        r"\bfrom\s+pathlib\b",
        r"\bfrom\s+shutil\b",

        # --- Write patterns ---
        r'\.to_csv\(',
        r'\.to_excel\(',
        r'\.to_parquet\(',
        r'open\s*\(',
        r'with\s+open\(',
        r'Path\(',
        r'os\.remove',
        r'os\.rmdir',
        r'shutil\.',
        r'pd\.to_pickle',

        # --- File system access ---
        r"\bopen\s*\(",
        r"\bos\.(remove|unlink|rmdir|mkdir|makedirs|rename|replace|chmod|chown)\b",
        r"\bshutil\.(rmtree|copy|copytree|move)\b",
        r"\bpathlib\.Path\s*\(",
        r"\.read\s*\(",
        r"\.write\s*\(",

        # --- Environment variables & sensitive data ---
        r"\bos\.environ\b",
        r"\bos\.getenv\s*\(",
        r"\bsys\.argv\b",
        r"\bsys\.path\b",

        # --- Code execution / reflection ---
        r"\beval\s*\(",
        r"\bexec\s*\(",
        r"\bcompile\s*\(",
        r"\b__import__\s*\(",
        r"\bglobals\s*\(",
        r"\blocals\s*\(",
        r"\bvars\s*\(",

        # --- Process / shell ---
        r"\bos\.system\s*\(",
        r"\bsubprocess\.",
        r"\bpopen\s*\("
        ]
    
    for pattern in forbidden_patterns:
        if re.search(pattern, code):
            return False  # Unsafe code detected
    
    return True  # Code is safe


# Need to run code with a timeout to prevent infinite loops or long-running code from hanging the system
import threading

class TimeoutError(Exception):
    """Raised when code execution times out."""
    pass

def run_with_timeout(code, safes, timeout=30):
    """
    Run code with a timeout using threading.
    
    Limitation: Won't kill infinite loops like while True,
    but will timeout on slow pandas/numpy/plotting operations.
    
    Args:
        code: String of Python code to execute
        safes: Dictionary of safe variables/functions
        timeout: Maximum time allowed - seconds (default 30)
    
    Returns:
        The result from safes.get("result")
    
    Raises:
        TimeoutError: If code runs longer than timeout
    """
    result_container = {"result": None, "error": None, "done": False}
    
    def execute_code():
        try:
            exec(code, safes)
            result_container["result"] = safes.get("result", None)
            result_container["done"] = True
        except Exception as e:
            import traceback
            result_container["error"] = traceback.format_exc()
            result_container["done"] = True
    
    thread = threading.Thread(target=execute_code)
    thread.daemon = True  # Thread dies when main program exits
    thread.start()
    thread.join(timeout=timeout)
    
    if not result_container["done"]:
        raise TimeoutError(f"Code execution timed out after {timeout} seconds")
    
    if result_container["error"]:
        raise Exception(result_container["error"])
    
    return result_container["result"]


### Logging setup
# When I create docker container - need to add ENV LOG_LEVEL=INFO
#ENV LOG_FILE=/app/logs/app.log
def setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_path = Path(os.getenv("LOG_FILE", "/app/logs/app.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.FileHandler(log_path, mode="a", encoding="utf-8")],
        force=True,
    )


if __name__ == "__main__":
    # Simple test of summarize_dataframe function
    #df = pd.read_csv("data/data_new.csv", sep="$")
    #df = pd.read_csv("../data/irae_data_cleaned.csv")
    #print(summarize_dataframe(df, max_rows=5))

    # Clean and export data file
    #cleaned_df = load_data()
    #cleaned_df.to_csv("data/irae_data_cleaned.csv", index=False)

    #df = pd.read_csv("../data/irae_data_cleaned.csv")
    #print(df.head(20))
    # Test that file operations are blocked

    # Tiny test for is_code_safe function
    assert is_code_safe("open('test.txt', 'w')") == False
    assert is_code_safe("import os") == False
    assert is_code_safe("os.remove('file.txt')") == False
    assert is_code_safe("__import__('os').system('rm file')") == False

    # Test that safe operations pass
    assert is_code_safe("result = df.groupby('column').sum()") == True
    assert is_code_safe("result = len(df)") == True


    # Test timeout with a slow operation (doesn't need imports)
    print("\n=== Testing timeout functionality ===")
    
    code = """
# Simulate slow operation without imports
total = 0
for i in range(100000000):  # Very slow loop
    total += i
result = total
    """
    safes = {"__builtins__": {"range": range, "len": len, "sum": sum, "list": list, "int": int}}  
    
    try:
        result = run_with_timeout(code, safes, timeout=2)
        print("❌ Should have timed out")
    except TimeoutError as e:
        print(f"✅ Timeout test passed: {e}")
    except Exception as e:
        print(f"❌ Different error: {e}")
    
    # Test that fast code works
    print("\n=== Testing fast code execution ===")
    code = "result = 2 + 2"
    safes = {"__builtins__": {}}
    
    try:
        result = run_with_timeout(code, safes, timeout=5)
        if result == 4:
            print(f"✅ Fast code test passed: result = {result}")
        else:
            print(f"❌ Wrong result: {result}")
    except Exception as e:
        print(f"❌ Fast code failed: {e}")
    
    
  
