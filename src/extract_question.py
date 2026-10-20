from datetime import date

import yaml
import csv
#/data/horse/ws/chmo273f-sparql_chain/sparql-chain/helper/hydroOntology/question_set
# Input and output files
date_now = date.today().strftime("%d_%m_%Y")
yaml_file = "/data/horse/ws/chmo273f-sparql_chain/sparql-chain/helper/hydroOntology/question_set/20260727_question_set/pre_training_22_03_2026_quadica.yaml"
csv_file = f"/data/horse/ws/chmo273f-sparql_chain/sparql-chain/helper/hydroOntology/question_set/20260727_question_set/{date_now}_questions_queries_quadica.csv"
# Read YAML
with open(yaml_file, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

# Write CSV
with open(csv_file, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f,  delimiter=";", quoting=csv.QUOTE_MINIMAL)

    # Header
    writer.writerow(["Hand-Written Question", "Query"])

    # Process each training example
    for item in data:
        question = ""
        query = ""

        for message in item.get("messages", []):
            role = message.get("role")

            if role == "user":
                question = message.get("content", "").strip()

            elif role == "assistant":
                query = message.get("content", "").strip()

        if question and query:
            writer.writerow([question, query])

print(f"CSV written to {csv_file}")