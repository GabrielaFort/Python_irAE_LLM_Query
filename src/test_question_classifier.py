from question_classifier import QuestionClassifier
from llm_client import LLMClient
from utils import llama3_llm   

# Instantiate question classifier
classifier = QuestionClassifier(llama3_llm())

questions = [
    "How many patients had colitis?",
    "Plot number of patients per tumor type.",
    "Females"
]

for q in questions:
    print(q, "->", classifier.classify(q))

    
