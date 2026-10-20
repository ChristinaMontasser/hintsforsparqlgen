import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional

from src.context_contracts import format_three_layer_context


@dataclass(frozen=True)
class LexiconMatch:
    section: str
    score: float
    entry: dict[str, Any]


class LexiconMapper:
    URI_KEYS = {
        "classes": "class_uri",
        "relations": "slot_uri",
        "entities": "uri",
    }

    def __init__(self, lexicons_path: str, fuzzy_threshold: float = 0.82):
        self.lexicons_path = Path(lexicons_path)
        self.fuzzy_threshold = fuzzy_threshold
        self.lexicons = self._load_lexicons()
        self.entries_by_section = {
            section: self.lexicons.get(section, [])
            for section in ("classes", "relations", "entities")
        }

    def _load_lexicons(self) -> dict[str, Any]:
        if not self.lexicons_path.exists():
            raise FileNotFoundError(f"Lexicons file not found: {self.lexicons_path}")
        return json.loads(self.lexicons_path.read_text(encoding="utf-8"))

    @staticmethod
    def normalize(text: str) -> str:
        text = (text or "").strip().lower()
        text = text.replace("_", " ").replace("-", " ")
        text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
        text = re.sub(r"[^\w:]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _surface_forms(self, entry: dict[str, Any], section: str) -> list[str]:
        uri_key = self.URI_KEYS.get(section)
        values = [entry.get("id"), entry.get("label"), entry.get(uri_key)]
        aliases = entry.get("aliases") or []
        if isinstance(aliases, list):
            values.extend(aliases)
        return [value for value in values if isinstance(value, str) and value.strip()]

    def _best_match_in_section(self, name: str, section: str) -> Optional[LexiconMatch]:
        raw_name = (name or "").strip()
        if not raw_name:
            return None

        normalized_name = self.normalize(raw_name)
        best_match = None

        for entry in self.entries_by_section.get(section, []):
            for surface in self._surface_forms(entry, section):
                if raw_name == surface:
                    return LexiconMatch(section=section, score=1.0, entry=entry)

                normalized_surface = self.normalize(surface)
                if normalized_name == normalized_surface:
                    return LexiconMatch(section=section, score=1.0, entry=entry)

                score = SequenceMatcher(None, normalized_name, normalized_surface).ratio()
                if score < self.fuzzy_threshold:
                    continue

                if best_match is None or score > best_match.score:
                    best_match = LexiconMatch(section=section, score=score, entry=entry)

        return best_match

    def match_name(self, name: str, preferred_sections: list[str]) -> Optional[LexiconMatch]:
        for section in preferred_sections:
            exact_match = self._best_match_in_section(name, section)
            if exact_match and exact_match.score == 1.0:
                return exact_match

        fallback_matches = [
            match for section in preferred_sections
            for match in [self._best_match_in_section(name, section)]
            if match is not None
        ]
        if not fallback_matches:
            return None

        return max(fallback_matches, key=lambda match: match.score)


class NERSchemaContextBuilder:
    def __init__(
        self,
        llm,
        schema_txt: str = None,
        system_prompt: str = None,
        user_prompt_template: str = None,
    ):
        self.llm = llm

        if schema_txt is not None:
            with open(schema_txt, "r", encoding="utf-8") as f:
                self.schema_txt = f.read()
        else:
            self.schema_txt = ""
        self.system_prompt =   system_prompt
        self.user_prompt_template = user_prompt_template
        self._context_cache: dict[str, str] = {}

    def get_context(self, question: str, question_id: Optional[str] = None) -> str:
        cache_key = f"{question_id or ''}::{question}"
        if cache_key in self._context_cache:
            return self._context_cache[cache_key]

        extraction = self.extract(question=question, question_id=question_id)
        # mapped = self.map_extraction(extraction)
        # context = self.format_context(extraction)
        self._context_cache[cache_key] = extraction
        return extraction

    def extract(self, question: str, question_id: Optional[str] = None) -> dict[str, Any]:
        prompt = self._build_prompt(question=question, question_id=question_id)
        response = self.llm(prompt)
        return self._parse_json_response(response)

    def _build_prompt(self, question: str, question_id: Optional[str]) -> str:
        question_id_value = question_id or "null"
        user_prompt = (
            self.user_prompt_template
            .replace("{question_id}", question_id_value)
            .replace("{question}", question)
            .replace("{schema}", self.schema_txt or "")
        )
        return f"{self.system_prompt}\n\n{user_prompt}"

    def _parse_json_response(self, response: str) -> dict[str, Any]:
        cleaned = (response or "").strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        json_block = self._extract_first_json_object(cleaned)
        if json_block is None:
            raise ValueError(f"NER model did not return valid JSON: {response}")
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

    def map_extraction(self, extraction: dict[str, Any]) -> dict[str, Any]:
        mapped_entities = [
            self._map_item(item, container_name="entities")
            for item in extraction.get("entities", [])
            if isinstance(item, dict)
        ]
        mapped_properties = [
            self._map_item(item, container_name="properties")
            for item in extraction.get("properties", [])
            if isinstance(item, dict)
        ]

        resolved_by_name: dict[str, list[dict[str, Any]]] = {}
        for item in mapped_entities + mapped_properties:
            source_name = self.mapper.normalize(item.get("source_name", ""))
            if source_name:
                resolved_by_name.setdefault(source_name, []).append(item)

        mapped_relationships = []
        for relationship in extraction.get("relationships", []):
            if not isinstance(relationship, dict):
                continue
            mapped_relationships.append({
                "source": relationship,
                "subject": self._resolve_relationship_endpoint(
                    relationship.get("subject"),
                    resolved_by_name,
                    default_sections=["classes", "entities"],
                ),
                "predicate": self._resolve_relationship_endpoint(
                    relationship.get("predicate"),
                    resolved_by_name,
                    default_sections=["relations"],
                ),
                "object": self._resolve_relationship_endpoint(
                    relationship.get("object"),
                    resolved_by_name,
                    default_sections=["classes", "entities", "relations"],
                ),
            })

        return {
            "question_id": extraction.get("question_id"),
            "intent": extraction.get("intent"),
            "entities": mapped_entities,
            "properties": mapped_properties,
            "relationships": mapped_relationships,
        }

    def _map_item(self, item: dict[str, Any], container_name: str) -> dict[str, Any]:
        name = item.get("name", "")
        item_type = item.get("type", "")
        preferred_sections = self._preferred_sections(
            container_name=container_name,
            item_type=item_type,
        )
        match = self.mapper.match_name(name, preferred_sections)
        return self._serialize_match(
            source_name=name,
            source_type=item_type,
            match=match,
        )

    def _resolve_relationship_endpoint(
        self,
        name: Any,
        resolved_by_name: dict[str, list[dict[str, Any]]],
        default_sections: list[str],
    ) -> dict[str, Any]:
        candidates = self._candidate_names(name)

        for source_name in candidates:
            normalized = self.mapper.normalize(source_name)
            if normalized in resolved_by_name:
                for section in default_sections:
                    for item in resolved_by_name[normalized]:
                        if item.get("section") == section:
                            return item

        match = None
        for source_name in candidates:
            match = self.mapper.match_name(source_name, default_sections)
            if match is not None:
                return self._serialize_match(
                    source_name=source_name,
                    source_type="relationship_component",
                    match=match,
                )

        return self._serialize_match(
            source_name=self._display_name(name),
            source_type="relationship_component",
            match=match,
        )

    @staticmethod
    def _candidate_names(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value] if value.strip() else [""]

        if isinstance(value, list):
            candidates = [
                item.strip()
                for item in value
                if isinstance(item, str) and item.strip()
            ]
            return candidates or [""]

        if value is None:
            return [""]

        return [str(value)]

    @staticmethod
    def _display_name(value: Any) -> str:
        if isinstance(value, list):
            parts = [item.strip() for item in value if isinstance(item, str) and item.strip()]
            return " | ".join(parts)
        if value is None:
            return ""
        return value if isinstance(value, str) else str(value)

    @staticmethod
    def _preferred_sections(container_name: str, item_type: str) -> list[str]:
        item_type_normalized = (item_type or "").strip().lower()

        if container_name == "properties":
            return ["relations"]

        if "class" in item_type_normalized:
            return ["classes"]

        if "relation" in item_type_normalized or "property" in item_type_normalized:
            return ["relations"]

        if "entity" in item_type_normalized or "instance" in item_type_normalized:
            return ["entities", "classes"]

        if container_name == "entities":
            return ["classes", "entities"]

        return ["classes", "relations", "entities"]

    def _serialize_match(
        self,
        source_name: str,
        source_type: str,
        match: Optional[LexiconMatch],
    ) -> dict[str, Any]:
        if match is None:
            return {
                "source_name": source_name,
                "source_type": source_type,
                "matched": False,
                "section": None,
                "score": None,
                "entry": None,
            }

        return {
            "source_name": source_name,
            "source_type": source_type,
            "matched": True,
            "section": match.section,
            "score": round(match.score, 4),
            "entry": match.entry,
        }

    def format_context(self, mapped: dict[str, Any]) -> str:
        lines = [
            "NER-derived schema hints:",
            "Use mapped URIs below when constructing SPARQL. Ignore unmatched free-text hints.",
        ]

        intent = mapped.get("intent")
        if isinstance(intent, str) and intent.strip():
            lines.append(f"Intent: {intent.strip()}")

        entity_items = mapped.get("entities", [])
        property_items = mapped.get("properties", [])
        sections = [
            ("Matched classes", entity_items, "classes"),
            ("Matched named entities", entity_items, "entities"),
            ("Matched relations", entity_items + property_items, "relations"),
        ]

        for title, items, expected_section in sections:
            matched_items = [
                item for item in items
                if item.get("matched") and item.get("section") == expected_section
            ]
            if not matched_items:
                continue

            lines.append("")
            lines.append(f"{title}:")
            for item in matched_items:
                entry = item["entry"]
                uri_key = self.mapper.URI_KEYS[expected_section]
                uri_value = entry.get(uri_key)
                label = entry.get("label") or entry.get("id") or item["source_name"]
                description = entry.get("description")
                line = f"- {label}"
                if uri_value:
                    line += f" -> {uri_value}"
                if description:
                    line += f" | {description}"
                if expected_section == "relations":
                    domain = entry.get("domain")
                    range_values = entry.get("range")
                    if domain:
                        line += f" | domain: {domain}"
                    if range_values:
                        line += f" | range: {range_values}"
                lines.append(line)

        matched_relationships = mapped.get("relationships", [])
        if matched_relationships:
            lines.append("")
            lines.append("Resolved relationship hints:")
            for relationship in matched_relationships:
                subject = self._format_endpoint(relationship["subject"])
                predicate = self._format_endpoint(relationship["predicate"])
                obj = self._format_endpoint(relationship["object"])
                lines.append(f"- {subject} -- {predicate} --> {obj}")

        unmatched = []
        for item in mapped.get("entities", []) + mapped.get("properties", []):
            if not item.get("matched"):
                unmatched.append(f"{item['source_type']}: {item['source_name']}")

        if unmatched:
            lines.append("")
            lines.append("Unmatched extracted hints:")
            lines.extend(f"- {item}" for item in unmatched)

        return format_three_layer_context(lines)

    @staticmethod
    def _format_endpoint(endpoint: dict[str, Any]) -> str:
        if endpoint.get("matched") and endpoint.get("entry"):
            entry = endpoint["entry"]
            uri = entry.get("class_uri") or entry.get("slot_uri") or entry.get("uri")
            return uri or entry.get("label") or endpoint.get("source_name", "?")
        return endpoint.get("source_name", "?")
