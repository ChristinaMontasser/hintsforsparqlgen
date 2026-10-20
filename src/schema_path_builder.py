import json
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional

from .ner_context_builder import LexiconMapper


@dataclass(frozen=True)
class ResolvedNode:
    node_id: str
    variable: str
    source_class: Optional[str]
    class_id: Optional[str]
    class_uri: Optional[str]
    kind: str
    matched: bool


@dataclass(frozen=True)
class ResolvedEdge:
    subject: str
    predicate: str
    obj: str
    source_predicate: str
    relation_id: Optional[str]
    relation_uri: Optional[str]
    domain: list[str]
    range: list[str]
    domain_ok: Optional[bool]
    range_ok: Optional[bool]
    required: bool
    reasoning_only: bool = False


class LexiconSchemaGraph:
    """Small graph view over the lexicon classes and relations."""

    RELATION_HINTS = {
        "included": {"hasPart", "isPartOf"},
        "includes": {"hasPart"},
        "include": {"hasPart"},
        "contains": {"hasPart"},
        "contained": {"isPartOf"},
        "part": {"hasPart", "isPartOf"},
        "name": {"name", "title"},
        "title": {"name", "title"},
        "author": {"author", "creator", "contributor"},
        "creator": {"creator", "author"},
        "publisher": {"publisher"},
        "license": {"license"},
        "url": {"url", "homepage", "landingPage", "accessURL", "downloadURL"},
        "website": {"url", "homepage", "hasWebsite"},
    }
    ANCHOR_STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "by",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "what",
        "which",
        "who",
        "with",
    }

    def __init__(self, lexicons_path: str, fuzzy_threshold: float = 0.82):
        self.lexicons_path = Path(lexicons_path)
        self.mapper = LexiconMapper(lexicons_path, fuzzy_threshold=fuzzy_threshold)
        self.lexicons = self.mapper.lexicons
        self.classes = self._entries_by_id("classes")
        self.relations = self._entries_by_id("relations")
        self.class_uri_to_id = {
            entry.get("class_uri"): class_id
            for class_id, entry in self.classes.items()
            if entry.get("class_uri")
        }

    @staticmethod
    def _as_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, str) and item.strip()]
        if isinstance(value, str) and value.strip():
            return [value]
        return []

    def _entries_by_id(self, section: str) -> dict[str, dict[str, Any]]:
        entries = self.lexicons.get(section, [])
        return {
            entry["id"]: entry
            for entry in entries
            if isinstance(entry, dict) and isinstance(entry.get("id"), str)
        }

    def class_ancestors(self, class_id: Optional[str]) -> set[str]:
        if not class_id or class_id not in self.classes:
            return set()

        ancestors = {class_id}
        queue = list(self._as_list(self.classes[class_id].get("parents")))
        while queue:
            parent = queue.pop(0)
            if parent in ancestors:
                continue
            ancestors.add(parent)
            parent_entry = self.classes.get(parent)
            if parent_entry:
                queue.extend(self._as_list(parent_entry.get("parents")))
        return ancestors

    def compatible_with(self, class_id: Optional[str], constraints: list[str]) -> Optional[bool]:
        if not constraints:
            return None
        if not class_id:
            return False
        return bool(self.class_ancestors(class_id).intersection(constraints))

    def resolve_class(self, value: Optional[str]) -> tuple[Optional[str], Optional[str], bool]:
        if not value:
            return None, None, False

        match = self.mapper.match_name(value, ["classes"])
        if not match or not match.entry:
            return None, None, False

        class_id = match.entry.get("id")
        return class_id, match.entry.get("class_uri"), bool(class_id)

    def resolve_relation(self, value: Optional[str]) -> tuple[Optional[str], Optional[str], list[str], list[str]]:
        if not value:
            return None, None, [], []

        match = self.mapper.match_name(value, ["relations"])
        if not match or not match.entry:
            return None, None, [], []

        entry = match.entry
        return (
            entry.get("id"),
            entry.get("slot_uri"),
            self._as_list(entry.get("domain")),
            self._as_list(entry.get("range")),
        )

    def candidate_context(self, question: str, max_classes: int, max_relations: int) -> str:
        class_candidates = self._rank_entries(question, "classes", "class_uri", max_classes)
        relation_candidates = self._rank_entries(question, "relations", "slot_uri", max_relations)
        relation_candidates = self._add_adjacent_relations(
            class_candidates,
            relation_candidates,
            max_relations,
        )

        lines = [
            "Candidate classes:",
        ]
        lines.extend(self._format_class_candidate(entry) for entry in class_candidates)
        lines.extend([
            "",
            "Candidate relations:",
        ])
        lines.extend(self._format_relation_candidate(entry) for entry in relation_candidates)
        return "\n".join(lines)

    def full_path_context(self) -> str:
        lines = [
            "SCHEMA:",
            "# Class hierarchy paths. These are reasoning-only links.",
        ]
        lines.extend(self._class_hierarchy_paths())
        lines.extend([
            "",
            "# Relation paths. Everything inside quotes is a relation.",
            "# The classes before the relation are valid heads; values after -> are valid tails.",
        ])
        lines.extend(self._relation_paths())
        return "\n".join(lines)

    def neighborhood_path_context(
        self,
        question: str,
        max_classes: int,
        max_relations: int,
    ) -> str:
        anchor_classes = self._anchor_classes(question, max_classes)
        anchor_relations = self._anchor_relations(question, max_relations)
        class_ids = {entry["id"] for entry in anchor_classes if entry.get("id")}

        if not class_ids:
            for relation in anchor_relations:
                class_ids.update(self._as_list(relation.get("domain")))
                class_ids.update(
                    class_id for class_id in self._as_list(relation.get("range"))
                    if class_id in self.classes
                )

        expanded_class_ids = set()
        for class_id in class_ids:
            expanded_class_ids.update(self.class_ancestors(class_id))

        relation_ids = {entry["id"] for entry in anchor_relations if entry.get("id")}
        for relation in self.lexicons.get("relations", []):
            relation_id = relation.get("id")
            if not relation_id or not relation.get("slot_uri"):
                continue
            domain = set(self._as_list(relation.get("domain")))
            range_values = {
                value for value in self._as_list(relation.get("range"))
                if value in self.classes
            }
            if (
                class_ids.intersection(domain | range_values)
                and self._touches_anchor_classes(domain | range_values, class_ids)
            ):
                relation_ids.add(relation_id)

        lines = [
            "SCHEMA:",
            "# Neighborhood schema paths selected from the full lexicon.",
            "# Class hierarchy paths. These are reasoning-only links.",
        ]
        lines.extend(self._class_hierarchy_paths_for(expanded_class_ids))
        lines.extend([
            "",
            "# Relation paths connected to the detected class/relation neighborhood.",
            "# Everything inside quotes is a relation.",
        ])
        lines.extend(self._relation_paths_for(relation_ids))
        return "\n".join(lines)

    def _anchor_classes(self, question: str, limit: int) -> list[dict[str, Any]]:
        question_text = self._normalized_for_anchor(question)
        question_tokens = self._anchor_tokens(question)
        scored = []

        for entry in self.lexicons.get("classes", []):
            if not isinstance(entry, dict) or not entry.get("class_uri"):
                continue
            score = self._surface_anchor_score(entry, "class_uri", question_text, question_tokens)
            if score:
                scored.append((score, entry))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [entry for _, entry in scored[:limit]]

    def _anchor_relations(self, question: str, limit: int) -> list[dict[str, Any]]:
        question_text = self._normalized_for_anchor(question)
        question_tokens = self._anchor_tokens(question)
        hinted_relation_ids = {
            relation_id
            for token in question_tokens
            for relation_id in self.RELATION_HINTS.get(token, set())
        }
        scored = []

        for entry in self.lexicons.get("relations", []):
            if not isinstance(entry, dict) or not entry.get("slot_uri"):
                continue
            score = self._surface_anchor_score(entry, "slot_uri", question_text, question_tokens)
            if entry.get("id") in hinted_relation_ids:
                score = max(score, 10)
            if score >= 6:
                scored.append((score, entry))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [entry for _, entry in scored[:limit]]

    def _surface_anchor_score(
        self,
        entry: dict[str, Any],
        uri_key: str,
        question_text: str,
        question_tokens: set[str],
    ) -> int:
        best_score = 0
        for surface in self._anchor_surfaces(entry, uri_key):
            surface_text = self._normalized_for_anchor(surface)
            surface_tokens = self._anchor_tokens(surface)
            if not surface_tokens:
                continue
            if surface_text and surface_text in question_text:
                best_score = max(best_score, 10 + len(surface_tokens))
            elif surface_tokens.issubset(question_tokens):
                best_score = max(best_score, 6 + len(surface_tokens))
            elif question_tokens.intersection(surface_tokens):
                best_score = max(best_score, len(question_tokens.intersection(surface_tokens)))
        return best_score

    def _anchor_surfaces(self, entry: dict[str, Any], uri_key: str) -> list[str]:
        values = [entry.get("id"), entry.get("label"), entry.get(uri_key)]
        aliases = entry.get("aliases") or []
        if isinstance(aliases, list):
            values.extend(aliases)

        surfaces = []
        for value in values:
            if not isinstance(value, str) or not value.strip():
                continue
            surfaces.append(value)
            surfaces.append(self._split_camel(value))
            if ":" in value:
                local_name = value.split(":", 1)[1]
                surfaces.append(local_name)
                surfaces.append(self._split_camel(local_name))
        return surfaces

    def _touches_anchor_classes(self, classes: set[str], anchor_class_ids: set[str]) -> bool:
        if not classes or not anchor_class_ids:
            return False
        for class_id in anchor_class_ids:
            if self.class_ancestors(class_id).intersection(classes):
                return True
        return False

    @staticmethod
    def _split_camel(value: str) -> str:
        value = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", value)
        return re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)

    def _normalized_for_anchor(self, value: str) -> str:
        return self.mapper.normalize(self._split_camel(value))

    def _anchor_tokens(self, value: str) -> set[str]:
        tokens = {
            token for token in self._normalized_for_anchor(value).split()
            if token not in self.ANCHOR_STOPWORDS
        }
        singularized = {
            token[:-1]
            for token in tokens
            if len(token) > 3 and token.endswith("s")
        }
        return tokens | singularized

    def _class_hierarchy_paths(self) -> list[str]:
        return self._class_hierarchy_paths_for(set(self.classes))

    def _class_hierarchy_paths_for(self, class_ids: set[str]) -> list[str]:
        lines = []
        for class_id in sorted(self.classes):
            if class_id not in class_ids:
                continue
            entry = self.classes[class_id]
            if not entry.get("class_uri"):
                continue
            for parent in self._as_list(entry.get("parents")):
                if parent in self.classes and parent in class_ids:
                    lines.append(f'"is_a": {class_id} -> {parent}')
        return lines or ['"is_a": None -> None']

    def _relation_paths(self) -> list[str]:
        return self._relation_paths_for(set(self.relations))

    def _relation_paths_for(self, relation_ids: set[str]) -> list[str]:
        lines = []
        for relation_id in sorted(self.relations):
            if relation_id not in relation_ids:
                continue
            entry = self.relations[relation_id]
            slot_uri = entry.get("slot_uri")
            if not slot_uri:
                continue

            heads = self._as_list(entry.get("domain")) or ["?x"]
            tails = self._as_list(entry.get("range")) or ["?literal"]
            head_text = ", ".join(heads)
            tail_text = ", ".join(tails)
            lines.append(f'"{slot_uri}": {head_text} -> {tail_text}')
        return lines or ['"?relation": ?x -> ?literal']

    def _rank_entries(self, question: str, section: str, uri_key: str, limit: int) -> list[dict[str, Any]]:
        normalized_question = self.mapper.normalize(question)
        question_tokens = set(normalized_question.split())
        scored = []

        for entry in self.lexicons.get(section, []):
            if not isinstance(entry, dict) or not entry.get(uri_key):
                continue
            surfaces = [
                entry.get("id"),
                entry.get("label"),
                entry.get("description"),
                " ".join(self._as_list(entry.get("aliases"))),
            ]
            text = self.mapper.normalize(" ".join(value for value in surfaces if isinstance(value, str)))
            tokens = set(text.split())
            overlap = len(question_tokens.intersection(tokens))
            ratio = SequenceMatcher(None, normalized_question, text).ratio()
            score = overlap + ratio
            if overlap or score >= 0.35:
                scored.append((score, entry))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [entry for _, entry in scored[:limit]]

    def _add_adjacent_relations(
        self,
        class_candidates: list[dict[str, Any]],
        relation_candidates: list[dict[str, Any]],
        max_relations: int,
    ) -> list[dict[str, Any]]:
        seen = {entry["id"] for entry in relation_candidates if entry.get("id")}
        candidate_class_ids = {
            ancestor
            for entry in class_candidates
            for ancestor in self.class_ancestors(entry.get("id"))
        }

        for relation in self.lexicons.get("relations", []):
            if len(relation_candidates) >= max_relations:
                break
            relation_id = relation.get("id")
            if not relation_id or relation_id in seen or not relation.get("slot_uri"):
                continue
            domain = set(self._as_list(relation.get("domain")))
            range_values = set(self._as_list(relation.get("range")))
            if candidate_class_ids.intersection(domain | range_values):
                relation_candidates.append(relation)
                seen.add(relation_id)

        return relation_candidates

    def _format_class_candidate(self, entry: dict[str, Any]) -> str:
        parents = ", ".join(self._as_list(entry.get("parents"))) or "None"
        label = entry.get("label") or entry.get("id")
        return f"- {entry.get('id')} ({label}) -> {entry.get('class_uri')} | parents: {parents}"

    def _format_relation_candidate(self, entry: dict[str, Any]) -> str:
        domain = ", ".join(self._as_list(entry.get("domain"))) or "Any/unknown"
        range_values = ", ".join(self._as_list(entry.get("range"))) or "Literal/unknown"
        label = entry.get("label") or entry.get("id")
        description = entry.get("description") or ""
        return (
            f"- {entry.get('id')} ({label}) -> {entry.get('slot_uri')} "
            f"| domain: {domain} | range: {range_values} | {description}"
        )


class SchemaPathContextBuilder:
    LITERAL_RANGE_TYPES = {
        "boolean",
        "date",
        "datetime",
        "decimal",
        "double",
        "float",
        "integer",
        "string",
        "time",
        "uri",
        "uriorcurie",
    }

    DEFAULT_SYSTEM_PROMPT = (
        "Your role is to find the reasoning path needed to answer the user query. "
        "A reasoning path is a set of uninterrupted chains in the form head --\"rel\"--> tail. "
        "Use only paths from the provided schema. "
        "Return only valid JSON with no explanation or markdown."
    )

    DEFAULT_USER_PROMPT_TEMPLATE = """Your role is to find the reasoning path needed to answer the user query.

A reasoning path is a set of chains: {head --"rel"--> tail}.
A path should not be interrupted.
Everything inside quotes is a relationship.
All entities before the relationship are valid heads of the relationship.
All entities after -> are valid tails of the relationship.
"is_a" is a reasoning-only relationship that means the head class can be treated as the tail class.

Rules:
- Use only classes and relationships from SCHEMA.
- Preserve relationship direction exactly.
- Use "is_a" paths when a class needs to satisfy the domain or range of a relationship.
- Return the smallest path needed to answer the question.
- Use class ids for node classes and relationship ids or URIs for predicates.

Return ONLY valid JSON using this schema:
{
  "question_id": "<question id or null>",
  "answer": {"variable": "?name", "description": "..."},
  "nodes": [
    {"id": "outer_article", "variable": "?article", "class": "Article", "kind": "class"},
    {"id": "lhb_article", "variable": "?lhbArticle", "class": "LHBArticle", "kind": "class"},
    {"id": "name", "variable": "?name", "kind": "literal"}
  ],
  "edges": [
    {"subject": "outer_article", "predicate": "hasPart", "object": "lhb_article", "required": true},
    {"subject": "lhb_article", "predicate": "name", "object": "name", "required": true}
  ],
  "intent": "..."
}

Question ID: {question_id}
Question: {question}

{schema_paths}
"""

    def __init__(
        self,
        llm,
        lexicons_path: str,
        fuzzy_threshold: float = 0.82,
        max_candidate_classes: int = 24,
        max_candidate_relations: int = 48,
        use_full_schema_paths: bool = True,
        schema_path_scope: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
    ):
        self.llm = llm
        self.graph = LexiconSchemaGraph(
            lexicons_path=lexicons_path,
            fuzzy_threshold=fuzzy_threshold,
        )
        self.max_candidate_classes = max_candidate_classes
        self.max_candidate_relations = max_candidate_relations
        self.schema_path_scope = schema_path_scope or (
            "full" if use_full_schema_paths else "neighborhood"
        )
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.user_prompt_template = user_prompt_template or self.DEFAULT_USER_PROMPT_TEMPLATE
        self._context_cache: dict[str, str] = {}
        self._artifact_cache: dict[str, dict[str, str]] = {}
        self._last_artifacts: dict[str, str] = {}
        self._last_prompt = ""
        self._last_schema_paths = ""
        self._last_raw_response = ""

    def get_context(self, question: str, question_id: Optional[str] = None) -> str:
        cache_key = f"{question_id or ''}::{question}"
        if cache_key in self._context_cache:
            self._last_artifacts = self._artifact_cache.get(cache_key, {})
            return self._context_cache[cache_key]

        path = self.extract(question=question, question_id=question_id)
        refined = self.refine_path(path)
        context = self.format_context(refined)
        artifacts = self._build_artifacts(
            raw_path=path,
            refined=refined,
            context=context,
        )
        self._context_cache[cache_key] = context
        self._artifact_cache[cache_key] = artifacts
        self._last_artifacts = artifacts
        return context

    def extract(self, question: str, question_id: Optional[str] = None) -> dict[str, Any]:
        prompt = self._build_prompt(question=question, question_id=question_id)
        response = self.llm(prompt)
        self._last_raw_response = response
        return self._parse_json_response(response)

    def _build_prompt(self, question: str, question_id: Optional[str]) -> str:
        schema_paths = self._schema_paths_for_prompt(question)
        self._last_schema_paths = schema_paths
        user_prompt = (
            self.user_prompt_template
            .replace("{question_id}", question_id or "null")
            .replace("{question}", question)
            .replace("{schema_paths}", schema_paths)
            .replace("{candidates}", schema_paths)
        )
        prompt = f"{self.system_prompt}\n\n{user_prompt}"
        self._last_prompt = prompt
        return prompt

    def get_last_artifacts(self) -> dict[str, str]:
        return dict(self._last_artifacts)

    def _build_artifacts(
        self,
        raw_path: dict[str, Any],
        refined: dict[str, Any],
        context: str,
    ) -> dict[str, str]:
        return {
            "schema_path_scope": self.schema_path_scope,
            "schema_path_schema": self._last_schema_paths,
            "schema_path_prompt": self._last_prompt,
            "schema_path_raw_response": self._last_raw_response,
            "schema_path_raw_json": json.dumps(raw_path, ensure_ascii=False),
            "schema_path_refined_json": json.dumps(
                self._serialize_refined(refined),
                ensure_ascii=False,
            ),
            "schema_path_context": context,
        }

    @staticmethod
    def _serialize_refined(refined: dict[str, Any]) -> dict[str, Any]:
        return {
            "question_id": refined.get("question_id"),
            "intent": refined.get("intent"),
            "answer": refined.get("answer"),
            "nodes": {
                key: asdict(value)
                for key, value in refined.get("nodes", {}).items()
            },
            "edges": [
                asdict(edge)
                for edge in refined.get("edges", [])
            ],
        }

    def _schema_paths_for_prompt(self, question: str) -> str:
        if self.schema_path_scope == "full":
            return self.graph.full_path_context()
        if self.schema_path_scope == "candidates":
            return self.graph.candidate_context(
                question=question,
                max_classes=self.max_candidate_classes,
                max_relations=self.max_candidate_relations,
            )
        if self.schema_path_scope != "neighborhood":
            raise ValueError(
                "schema_path_scope must be one of: full, neighborhood, candidates"
            )
        return self.graph.neighborhood_path_context(
            question=question,
            max_classes=self.max_candidate_classes,
            max_relations=self.max_candidate_relations,
        )

    def _parse_json_response(self, response: str) -> dict[str, Any]:
        cleaned = (response or "").strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            json_block = self._extract_first_json_object(cleaned)
            if json_block is None:
                raise ValueError(f"Schema-path model did not return valid JSON: {response}")
            return json.loads(json_block)

    @staticmethod
    def _extract_first_json_object(text: str) -> Optional[str]:
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        for index in range(start, len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start:index + 1]
        return None

    def refine_path(self, path: dict[str, Any]) -> dict[str, Any]:
        nodes = self._resolve_nodes(path.get("nodes", []))
        edges = self._resolve_edges(path.get("edges", []), nodes)
        return {
            "question_id": path.get("question_id"),
            "intent": path.get("intent"),
            "answer": path.get("answer"),
            "nodes": nodes,
            "edges": edges,
        }

    def _resolve_nodes(self, raw_nodes: Any) -> dict[str, ResolvedNode]:
        resolved = {}
        if not isinstance(raw_nodes, list):
            return resolved

        for raw_node in raw_nodes:
            if not isinstance(raw_node, dict):
                continue
            node_id = raw_node.get("id") or raw_node.get("variable")
            if not isinstance(node_id, str) or not node_id.strip():
                continue
            variable = self._safe_variable(raw_node.get("variable"), fallback=node_id)
            source_class = raw_node.get("class")
            class_id, class_uri, matched = self.graph.resolve_class(source_class)
            resolved[node_id] = ResolvedNode(
                node_id=node_id,
                variable=variable,
                source_class=source_class,
                class_id=class_id,
                class_uri=class_uri,
                kind=raw_node.get("kind") or "class",
                matched=matched,
            )
        return resolved

    def _resolve_edges(self, raw_edges: Any, nodes: dict[str, ResolvedNode]) -> list[ResolvedEdge]:
        resolved = []
        if not isinstance(raw_edges, list):
            return resolved

        for raw_edge in raw_edges:
            if not isinstance(raw_edge, dict):
                continue
            subject_id = raw_edge.get("subject")
            object_id = raw_edge.get("object")
            source_predicate = raw_edge.get("predicate")
            reasoning_only = str(source_predicate or "").strip() == "is_a"
            relation_id, relation_uri, domain, range_values = self.graph.resolve_relation(source_predicate)
            subject_node = nodes.get(subject_id)
            object_node = nodes.get(object_id)
            if reasoning_only:
                domain_ok = None
                range_ok = None
            else:
                domain_ok = self.graph.compatible_with(
                    subject_node.class_id if subject_node else None,
                    domain,
                )
                range_ok = self._range_ok(object_node, range_values)
            resolved.append(ResolvedEdge(
                subject=subject_node.variable if subject_node else self._safe_variable(subject_id, "subject"),
                predicate="is_a" if reasoning_only else relation_uri or str(source_predicate or "?predicate"),
                obj=object_node.variable if object_node else self._safe_variable(object_id, "object"),
                source_predicate=str(source_predicate or ""),
                relation_id=relation_id,
                relation_uri=relation_uri,
                domain=domain,
                range=range_values,
                domain_ok=domain_ok,
                range_ok=range_ok,
                required=bool(raw_edge.get("required", True)),
                reasoning_only=reasoning_only,
            ))
        return resolved

    def _range_ok(self, object_node: Optional[ResolvedNode], range_values: list[str]) -> Optional[bool]:
        if not range_values:
            return None
        if object_node and object_node.kind == "literal":
            return all(value.lower() in self.LITERAL_RANGE_TYPES for value in range_values)
        return self.graph.compatible_with(object_node.class_id if object_node else None, range_values)

    @staticmethod
    def _safe_variable(value: Any, fallback: str) -> str:
        if isinstance(value, str) and value.startswith("?"):
            return value
        base = re.sub(r"\W+", "_", str(value or fallback)).strip("_") or fallback
        return f"?{base}"

    def format_context(self, refined: dict[str, Any]) -> str:
        lines = [
            "Schema-path-derived graph plan:",
            "Use this graph plan as the preferred structure for the SPARQL query.",
            "Do not invent predicates or classes outside the resolved URIs below.",
        ]

        intent = refined.get("intent")
        if isinstance(intent, str) and intent.strip():
            lines.append(f"Intent: {intent.strip()}")

        answer = refined.get("answer")
        if isinstance(answer, dict):
            variable = answer.get("variable")
            description = answer.get("description")
            if variable or description:
                lines.append(f"Answer target: {variable or '?answer'} ({description or 'no description'})")

        nodes = refined.get("nodes", {})
        edges = refined.get("edges", [])
        reasoning_edges = [edge for edge in edges if edge.reasoning_only]
        relation_edges = [edge for edge in edges if not edge.reasoning_only]

        active_variables = {
            variable
            for edge in relation_edges
            for variable in (edge.subject, edge.obj)
        }
        class_patterns = self._class_patterns(nodes, active_variables)
        if class_patterns:
            lines.extend(["", "Resolved class patterns:"])
            lines.extend(f"- {pattern}" for pattern in class_patterns)

        if reasoning_edges:
            lines.extend(["", "Reasoning-only hierarchy paths:"])
            for edge in reasoning_edges:
                lines.append(f"- {edge.subject} --is_a--> {edge.obj}")

        if relation_edges:
            lines.extend(["", "Resolved relation patterns:"])
            for edge in relation_edges:
                prefix = "" if edge.required else "OPTIONAL "
                lines.append(f"- {prefix}{edge.subject} {edge.predicate} {edge.obj}")

        validation_lines = self._validation_lines(relation_edges)
        if validation_lines:
            lines.extend(["", "Domain/range checks:"])
            lines.extend(validation_lines)

        return "\n".join(lines)

    @staticmethod
    def _class_patterns(nodes: dict[str, ResolvedNode], active_variables: set[str]) -> list[str]:
        patterns = []
        for node in nodes.values():
            if node.class_uri and node.variable in active_variables:
                patterns.append(f"{node.variable} rdf:type {node.class_uri}")
        return patterns

    @staticmethod
    def _validation_lines(edges: list[ResolvedEdge]) -> list[str]:
        lines = []
        for edge in edges:
            domain = ", ".join(edge.domain) or "Any/unknown"
            range_values = ", ".join(edge.range) or "Literal/unknown"
            domain_status = "ok" if edge.domain_ok is not False else "not confirmed"
            range_status = "ok" if edge.range_ok is not False else "not confirmed"
            lines.append(
                f"- {edge.source_predicate}: domain {domain} ({domain_status}); "
                f"range {range_values} ({range_status})"
            )
        return lines
