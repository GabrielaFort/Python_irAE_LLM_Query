import pandas as pd
from manager import Manager

df = pd.read_csv("../data/irae_data_cleaned.csv")
m = Manager(df)

questions = [
    "Give me all lung cancer patients with more than two iraes.",
    "How many patients had more than two iraes?"
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