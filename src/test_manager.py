import pandas as pd
from manager import Manager

df = pd.read_csv("../data/irae_data_cleaned.csv")
m = Manager(df)

questions = [
    "How many patients had colitis?",
    "Plot number of patients per tumor type.",
    "Females",
    "All patients",
    "Show me a histogram of age distribution.",
    "What is the average age of patients with pneumonitis?"
]

for q in questions:
    res = m.process_question(q)
    print("Question:", q)
    print("Result type:", res["type"])
    print("Code:\n", res["code"])
    if res["type"] == "plot":
        print("(Plot generated)")
    else:
        print(res["data"])
    print("-----\n")