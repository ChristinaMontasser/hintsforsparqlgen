## load the questions 
## instantiate similarity measure class, especially query similarity measure

## camels --> pairwise similarity measure
## lamah --> pairwise similarity measure
## quadica --> pairwise similarity measure
## camels-lamah
## camels-quadica
## lamah-quadica

from itertools import combinations
import pandas as pd
import argparse
from pathlib import Path
from itertools import combinations
from src.validate_query import QuerySimilarity
import ast
import re

def main():
    parser = argparse.ArgumentParser(description="Pairwise query similarity comparison")
    parser.add_argument(
        "--output-dir",
        default="pairwise_query_similarity",
        help="Output directory",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset files
    dataset_files = {
        "camels": "helper/hydroOntology/question_set/20260727_question_set/28_07_2026_questions_queries_camels_gb.csv",
        "lamah": "helper/hydroOntology/question_set/20260727_question_set/27_07_2026_questions_queries_lamah_ce.csv",
        "quadica": "helper/hydroOntology/question_set/20260727_question_set/28_07_2026_questions_queries_quadica.csv",
    }

    # Load datasets
    datasets = {
        # "camels": pd.read_csv(dataset_files["camels"], sep=";"),
        # "camels_": pd.read_csv(dataset_files["camels"], sep=";"),

        # "lamah": pd.read_csv(dataset_files["lamah"], sep=";"),
        # "lamah_": pd.read_csv(dataset_files["lamah"], sep=";"),

        "quadica": pd.read_csv(dataset_files["quadica"], sep=";"),
        "quadica_": pd.read_csv(dataset_files["quadica"], sep=";"),

    }

    similarity_measure = QuerySimilarity()

    for ds1, ds2 in combinations(datasets.keys(), 2):
        print(ds1)
        print(ds2)
        results = []

        queries1 = datasets[ds1]["Query"]
        queries2 = datasets[ds2]["Query"]

        for i, query1 in enumerate(queries1):
            for j, query2 in enumerate(queries2):

                similarities = similarity_measure.measure_similarities(query1, query2)

                results.append({
                    "dataset_1": ds1,
                    "query_1_idx": i,
                    "query_1": query1,
                    "dataset_2": ds2,
                    "query_2_idx": j,
                    "query_2": query2,
                    **similarities
                })

        df = pd.DataFrame(results)
        df.to_csv(output_dir / f"{ds1}_{ds2}_query_similarity.csv", index=False)

#main()

### Calulate average for  all 


input_dir = Path("pairwise_query_similarity")
output_file = input_dir / "pairwise_similarity_summary.csv"

def get_structure_components(value):
    """Extract individual structure similarity components."""
    value = re.sub(r"np\.float32\((.*?)\)", r"\1", value)
    similarity_dict = ast.literal_eval(value)

    return pd.Series({
        "string_similarity": similarity_dict["string_similarity"],
        "graph_similarity": similarity_dict["graph_similarity"],
        "semantic_similarity": similarity_dict["semantic_similarity"],
    })


def get_language_similarity(value):
    """Extract frequency_similarity from language similarity dictionary."""
    value = re.sub(r"np\.float32\((.*?)\)", r"\1", value)
    return ast.literal_eval(value)["frequency_similarity"]


summary = []

for file in sorted(input_dir.glob("*similarity.csv")):

    df = pd.read_csv(file)

    # Extract structure components
    structure_components = df["structure_similarity"].apply(
        get_structure_components
    )

    df = pd.concat([df, structure_components], axis=1)

    # Extract language similarity
    df["language_similarity_value"] = df["language_similarity"].apply(
        get_language_similarity
    )

    string_avg = df["string_similarity"].mean()
    graph_avg = df["graph_similarity"].mean()
    semantic_avg = df["semantic_similarity"].mean()

    summary.append({
        "dataset_1": df["dataset_1"].iloc[0],
        "dataset_2": df["dataset_2"].iloc[0],

        "string_similarity avg": string_avg,
        "graph_similarity avg": graph_avg,
        "semantic_similarity avg": semantic_avg,

        # average of the three structure components
        "structure_similarity avg": (
            string_avg + graph_avg + semantic_avg
        ) / 3,

        "content_similarity avg": pd.to_numeric(
            df["content_similarity"], errors="coerce"
        ).mean(),

        "language_similarity avg": df["language_similarity_value"].mean(),
    })


summary_df = pd.DataFrame(summary)

summary_df.to_csv(output_file, index=False)

print(summary_df)
print(f"Saved to {output_file}")