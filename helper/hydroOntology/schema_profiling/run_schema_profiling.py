#!/usr/bin/env python3
"""Generate a graph schema profile once for a given RDF graph.

This script converts the downloaded schema-profiling notebook into a single
executable pipeline. It:
1. queries all classes in the selected graph,
2. queries all predicates in the selected graph,
3. infers subject/object ranges for each predicate,
4. saves both CSV and compact TXT schema snapshots.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from ruamel.yaml import YAML


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in (current.parents[3], current.parents[2], current.parents[1], current.parents[0]):
        if (candidate / "src" / "validate_query.py").exists():
            return candidate
    raise FileNotFoundError("Could not find project root containing src/validate_query.py")


def ensure_project_root() -> Path:
    project_root = find_project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    os.chdir(project_root)
    return project_root


def uri_to_qname(uri: str, prefix_map: dict[str, str]) -> str | None:
    if uri is None:
        return None
    for namespace, prefix in prefix_map.items():
        if str(uri).startswith(namespace):
            return f"{prefix}:{str(uri)[len(namespace):]}"
    return str(uri)


def is_uri(value: object) -> bool:
    return isinstance(value, str) and (value.startswith("http://") or value.startswith("https://"))


def validate_query(query: str, endpoint: str):
    from src.validate_query import validate_query as _validate_query

    return _validate_query(query, endpoint)


DEFAULT_PREFIXES_FILE = Path(__file__).with_name("prefixes.yaml")


def load_prefix_map(prefixes_file: Path) -> dict[str, str]:
    if not prefixes_file.is_file():
        raise FileNotFoundError(f"Prefix file not found: {prefixes_file}")
    data = YAML(typ="safe").load(prefixes_file.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(
        isinstance(namespace, str)
        and isinstance(prefix, str)
        and namespace
        and prefix
        for namespace, prefix in data.items()
    ):
        raise ValueError(f"Invalid prefix mapping: {prefixes_file}")
    return data



def run_class_extraction(graph: str, endpoint: str, output_dir: Path):
    graph_uri = graph.strip("/")
    filename_graph = graph_uri.replace("/", "_")
    query = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

    SELECT ?class
    WHERE {{
      GRAPH <{graph_uri}> {{
        ?s rdf:type ?class .
      }}
    }}
    GROUP BY ?class
    """

    result = validate_query(query, endpoint)
    if not result.get("success"):
        raise RuntimeError(f"Class extraction failed for {graph}: {result}")

    bindings = result["data"]["results"]["bindings"]
    classes_df = pd.DataFrame(bindings)
    classes_df["class"] = classes_df["class"].apply(lambda x: x["value"])

    filename = output_dir / f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}_{filename_graph}_hydroClasses.csv"
    classes_df.to_csv(filename, index=False)
    print(f"Saved classes: {filename}")
    return classes_df


def run_predicate_extraction(graph: str, endpoint: str, output_dir: Path):
    graph_uri = graph.strip("/")
    filename_graph = graph_uri.replace("/", "_")
    query = f"""
    SELECT ?p
    WHERE {{
      GRAPH <{graph_uri}> {{
        ?s ?p ?o .
      }}
    }}
    GROUP BY ?p
    """

    result = validate_query(query, endpoint)
    if not result.get("success"):
        raise RuntimeError(f"Predicate extraction failed for {graph}: {result}")

    bindings = result["data"]["results"]["bindings"]
    predicates_df = pd.DataFrame(bindings)
    predicates_df["p"] = predicates_df["p"].apply(lambda x: x["value"])

    filename = output_dir / f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}_{filename_graph}_hydroPredicates.csv"
    predicates_df.to_csv(filename, index=False)
    print(f"Saved predicates: {filename}")
    return predicates_df


def build_predicate_schema(graph: str, endpoint: str, output_dir: Path):
    graph_uri = graph.strip("/")
    filename_graph = graph_uri.replace("/", "_")
    predicates_path = output_dir / next(
        p.name for p in sorted(output_dir.glob(f"*{filename_graph}_hydroPredicates.csv"))
    )
    if not predicates_path.exists():
        raise FileNotFoundError(f"Predicates file not found: {predicates_path}")

    predicates_df = pd.read_csv(predicates_path)
    rows: list[dict] = []
    rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

    for uri in predicates_df["p"]:
        print(f"Processing: {uri}")

        if uri == rdf_type:
            type_query = f"""
                SELECT ?class (COUNT(*) AS ?count)
                WHERE {{
                  GRAPH <{graph_uri}> {{
                    ?s rdf:type ?class .
                  }}
                }}
                GROUP BY ?class
                ORDER BY DESC(?count)
                LIMIT 100
            """
            type_result = validate_query(type_query, endpoint)
            if type_result.get("success"):
                for binding in type_result["data"]["results"]["bindings"]:
                    rows.append({
                        "predicate": "rdf:type",
                        "subject_class": None,
                        "object": uri_to_qname(binding["class"]["value"], PREFIX_MAP),
                        "property_type": "TypeAssertion",
                    })
            continue

        subject_query = f"""
            SELECT ?sClass (COUNT(*) AS ?count)
            WHERE {{
              GRAPH <{graph_uri}> {{
                ?s <{uri}> ?o .
                ?s a ?sClass .
              }}
            }}
            GROUP BY ?sClass
            ORDER BY DESC(?count)
            LIMIT 20
        """
        subject_result = validate_query(subject_query, endpoint)
        subject_classes = []
        if subject_result.get("success"):
            subject_classes = [b["sClass"]["value"] for b in subject_result["data"]["results"]["bindings"]]

        object_query = f"""
            SELECT ?oClass (COUNT(*) AS ?count)
            WHERE {{
              GRAPH <{graph_uri}> {{
                ?s <{uri}> ?o .
                ?o a ?oClass .
              }}
            }}
            GROUP BY ?oClass
            ORDER BY DESC(?count)
            LIMIT 50
        """
        object_result = validate_query(object_query, endpoint)
        object_classes = []
        if object_result.get("success"):
            object_classes = [b["oClass"]["value"] for b in object_result["data"]["results"]["bindings"]]

        datatype_query = f"""
            SELECT ?datatype (COUNT(*) AS ?count)
            WHERE {{
              GRAPH <{graph_uri}> {{
                ?s <{uri}> ?o .
                FILTER(isLiteral(?o))
                BIND(DATATYPE(?o) AS ?datatype)
              }}
            }}
            GROUP BY ?datatype
            ORDER BY DESC(?count)
            LIMIT 20
        """
        datatype_result = validate_query(datatype_query, endpoint)
        datatypes = []
        if datatype_result.get("success"):
            datatypes = [b["datatype"]["value"] if "datatype" in b else "Literal" for b in datatype_result["data"]["results"]["bindings"]]

        sample_query = f"""
            SELECT DISTINCT ?o
            WHERE {{
              GRAPH <{graph_uri}> {{
                ?s <{uri}> ?o .
              }}
            }}
            LIMIT 50
        """
        sample_result = validate_query(sample_query, endpoint)
        sample_objects = []
        if sample_result.get("success"):
            sample_objects = [b["o"]["value"] for b in sample_result["data"]["results"]["bindings"]]

        if len(datatypes) > 0:
            property_type = "DatatypeProperty"
            object_ranges = [uri_to_qname(x, PREFIX_MAP) for x in datatypes]
        elif len(object_classes) > 0:
            property_type = "ObjectProperty"
            object_ranges = [uri_to_qname(x, PREFIX_MAP) for x in object_classes]
        elif sample_objects and all(is_uri(o) for o in sample_objects):
            property_type = "ObjectProperty"
            object_ranges = sorted({uri_to_qname(o, PREFIX_MAP) for o in sample_objects})
        else:
            property_type = "Unknown"
            object_ranges = ["?x"]

        subject_qnames = [uri_to_qname(x, PREFIX_MAP) for x in subject_classes]
        if len(subject_qnames) == 0:
            subject_qnames = [None]

        for subj in subject_qnames:
            for obj in object_ranges:
                rows.append({
                    "predicate": uri_to_qname(uri, PREFIX_MAP),
                    "subject_class": subj,
                    "object": obj,
                    "property_type": property_type,
                })

    schema_df = pd.DataFrame(rows)
    csv_file = output_dir / f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}_{filename_graph}_predicate_schema.csv"
    schema_df.to_csv(csv_file, index=False)
    print(f"Saved predicate schema: {csv_file}")
    return schema_df


def predicate_schema_csv_to_txt(csv_file: Path, txt_file: Path):
    df = pd.read_csv(csv_file)
    lines = []

    for _, row in df.iterrows():
        predicate = row["predicate"] if pd.notna(row["predicate"]) else "?predicate"
        subject = row["subject_class"] if pd.notna(row["subject_class"]) and str(row["subject_class"]).strip() not in {"", "nan"} else "?instance"
        obj = row["object"] if pd.notna(row["object"]) and str(row["object"]).strip() not in {"", "nan"} else "?x"
        lines.append(f"{predicate}: {subject} -> {obj}")

    txt_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved TXT schema: {txt_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Profile a graph schema once and save CSV/TXT outputs.")
    parser.add_argument("--graph", required=True, help="Complete graph path including its prefix")
    parser.add_argument("--endpoint", required=True, help="SPARQL endpoint URL")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write schema output files. Defaults to helper/hydroOntology/schema_profiling",
    )
    parser.add_argument("--prefixes-file", type=Path, default=DEFAULT_PREFIXES_FILE, help="YAML file containing URI-to-prefix mappings")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = ensure_project_root()

    output_dir = Path(args.output_dir) if args.output_dir else project_root / "helper" / "hydroOntology" / "schema_profiling"
    output_dir.mkdir(parents=True, exist_ok=True)

    graph = args.graph
    endpoint = args.endpoint
    global PREFIX_MAP
    PREFIX_MAP = load_prefix_map(args.prefixes_file)

    print(f"Project root: {project_root}")
    print(f"Graph: {graph}")
    print(f"Endpoint: {endpoint}")
    print(f"Output directory: {output_dir}")

    run_class_extraction(graph, endpoint, output_dir)
    run_predicate_extraction(graph, endpoint, output_dir)
    build_predicate_schema(graph, endpoint, output_dir)

    filename_graph = graph.strip("/").replace("/", "_")
    csv_file = output_dir / next(
        p.name for p in sorted(output_dir.glob(f"*{filename_graph}_predicate_schema.csv"))
    )
    txt_file = output_dir / f"{csv_file.stem}.txt"
    predicate_schema_csv_to_txt(csv_file, txt_file)

    print(f"\nCompleted for graph: {graph}")
    print(f"Schema CSV: {csv_file}")
    print(f"Schema TXT: {txt_file}")


if __name__ == "__main__":
    main()
