from question_classifier import QuestionClassifier
from llm_client import LLMClient
from utils import stats_llm   

# Instantiate question classifier
classifier = QuestionClassifier(stats_llm())

questions = [
    "How many patients had colitis?",
    "Plot number of patients per tumor type.",
    "Females",
    "Show me a histogram of age distribution.",
    "What is the average survival time?",
    "Is there a significant difference in time to onset between males and females?",
    "Is there a difference in irae types between age groups?"

]

for q in questions:
    print(q, "->", classifier.classify(q))

    
