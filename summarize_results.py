import os
import re
import ast
import json
import pandas as pd

# Directory containing experiment results
EXPS_DIR = "exps/"

def summarize_results(output_csv="results_summary.csv"):
    """
    Reads results from the 'exps/{exp_name}/results/' directories, extracts ROUGE scores,
    saves them to a CSV file, and displays them in Markdown format.

    Args:
        output_csv (str): The name of the output CSV file.
    """
    experiments = []

    # Iterate through experiment directories
    for exp_name in os.listdir(EXPS_DIR):
        results_file = os.path.join(EXPS_DIR, exp_name, "results", "test_result_0shot.txt")

        if os.path.exists(results_file):
            with open(results_file, "r", encoding="utf-8") as f:
                content = f.read()

                # Extract ROUGE scores using regex
                rouge_match = re.search(r"Rouge:\s*(\{.+\})", content)
                if rouge_match:
                    rouge_scores = ast.literal_eval(rouge_match.group(1))  # Convert string to dictionary
                else:
                    rouge_scores = {}

                # Store results
                experiments.append({
                    "Experiment": exp_name,
                    "ROUGE-1": rouge_scores.get("rouge1", None),
                    "ROUGE-2": rouge_scores.get("rouge2", None),
                    "ROUGE-L": rouge_scores.get("rougeL", None),
                    "ROUGE-Lsum": rouge_scores.get("rougeLsum", None),
                })

    # Convert to Pandas DataFrame
    df_results = pd.DataFrame(experiments)

    # Save to CSV
    df_results.to_csv(output_csv, index=False)
    print(f"✅ Results saved to {output_csv}")

    # Show Markdown table
    markdown_table = df_results.to_markdown(index=False)
    print("\n### 📊 Experiment Results\n")
    print(markdown_table)

    return df_results  # Return DataFrame for further analysis

# # Run function
# results = summarize_results()
