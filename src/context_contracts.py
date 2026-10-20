from typing import Iterable


HYDRO_DATASET_CONTRACT_LINES = [ #How this particular RDF store is physically/logically organized.
    "Query hydro data from GRAPH <hydroturtle/lamah_ce> or FROM <hydroturtle/lamah_ce>.",
    "Use canonical hydro prefixes and never invent example.org IRIs or prefixes.",
    "Station names are represented with schema:name.",
    "Observation times are represented with sosa:resultTime.",
    "Numeric observation values are retrieved via sosa:hasResult and qudt:numericValue.",
    "Geometries are retrieved via geo:hasGeometry and geo:asWKT.",
]

#Ontology Contract #What concepts and relations are semantically valid

HYDRO_QUERY_PLANNING_CONTRACT_LINES = [ #In which way queries is preferred to be constructed
    "Build the query only from ontology-supported paths and resolved URIs.",
    "Prefer canonical dataset access patterns over invented shortcut predicates.",
    "If a measurement value is required, prefer the observation-result-value chain when applicable.",
    "If a temporal constraint is required, attach it through sosa:resultTime when applicable.",
    "Do not invent classes, predicates, or named entities that are not justified by the ontology contract.",
]


def format_three_layer_context(
    ontology_contract_lines: Iterable[str],
    dataset_contract_lines: Iterable[str] | None = None,
    planning_contract_lines: Iterable[str] | None = None,
) -> str:
    dataset_lines = list(dataset_contract_lines or HYDRO_DATASET_CONTRACT_LINES)
    ontology_lines = [line for line in ontology_contract_lines if line is not None]
    planning_lines = list(planning_contract_lines or HYDRO_QUERY_PLANNING_CONTRACT_LINES)

    sections: list[str] = ["Dataset contract:"]
    sections.extend(f"- {line}" for line in dataset_lines)
    sections.append("")
    sections.append("Ontology contract:")
    sections.extend(ontology_lines)
    sections.append("")
    sections.append("Query planning contract:")
    sections.extend(f"- {line}" for line in planning_lines)
    return "\n".join(sections)
