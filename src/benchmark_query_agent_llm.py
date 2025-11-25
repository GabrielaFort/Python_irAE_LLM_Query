import pandas as pd
import os 
import time

from src.llm_client import LLMClient
from src.utils import load_data, summarize_dataframe

# Benchmarking script for Query Agent LLM on irAE dataset using LLM as a judge metric
# Already have LLM code and gold code from first benchmarking round

def benchmark_query_agent(llm_code, gold_code, data_preview, question, model="qwen3-coder:480b-cloud"):
    # Initialize LLM client
    llm_client = LLMClient(model=model,
                           api_url="https://ollama.com",
                           temperature=0,
                           api_key=os.getenv("OLLAMA_API_KEY"))
    # Generate LLM result
    prompt = f"""
You are an expert Python code reviewer.

Data preview:
{data_preview}

Question:
{question}

Compare the two Python code snippets below and rate how *functionally equivalent* they are on a scale of 0–10.
Functional equivalence means both correctly answer the question above on the given data,
even if implemented differently.

Be tolerant of:
- Equivalent filters, counts, or groupings expressed differently
- Equivalent error or invalid-query messages
- Minor formatting or sorting differences
- Multi-step vs single-step implementations producing the same result

Return only a single integer 0–10 on the first line of your response.

--- GOLD CODE ---
{gold_code}

--- LLM CODE ---
{llm_code}
"""
    
    llm_response = llm_client.generate(prompt)
    llm_response = llm_response.strip()

    # Parse response for score
    # Extract first occurence of 0 or 1
    import re
    match = re.search(r"\b(10|[0-9])\b", llm_response)
    if match:
        score = int(match.group(0))
    else:
        print(f"Warning: No valid score found in LLM response: {llm_response}")
        score = 0

    return score, llm_response


if __name__ == "__main__":
    # Load benchmark cases
    testing_model = "qwen3-coder:480b-cloud"
    input_file = f"data/benchmark_query_agent_results_{testing_model}.csv"
    df_benchmark = pd.read_csv(input_file)

    data_preview = load_data()
    data_preview = summarize_dataframe(data_preview, max_rows=10)

    model = "deepseek-v3.1:671b-cloud"
    results = []

    for _, row in df_benchmark.iterrows():
        llm_code = row["llm_code"]
        gold_code = row["gold_code"]
        question = row["question"]

        score, raw_response = benchmark_query_agent(llm_code, gold_code, data_preview, question, model=model)

        results.append({
            "question": question,
            "llm_code": llm_code,
            "gold_code": gold_code,
            "judge_model": model,
            "score": score,
            "raw_response": raw_response
        })

        time.sleep(20) # Pause between requests to avoid rate limits

    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    df_results["equivalent"] = (df_results["score"] >= 8).astype(int)

    # Compute summary stats
    accuracy = 100 * df_results["equivalent"].mean()
    print(f"Functional equivalence ≥8/10 accuracy: {accuracy:.1f}%")

    # Save to CSV
    output_file = f"data/benchmark_query_agent_judged_{testing_model}.csv"
    df_results.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
