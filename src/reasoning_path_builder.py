import json
import re
from pathlib import Path
from typing import Optional

import yaml


class ReasoningPathBuilder:
    DEFAULT_SYSTEM_PROMPT = (
        "Your role is to find the reasoning path needed to answer the user query. "
        "Return only the requested reasoning path text."
    )

    DEFAULT_USER_PROMPT_TEMPLATE = """Your role is to find the reasoning path to answer the user query.

A reasoning path is a set of chains {head -> rel -> tail}.
A path should not be interrupted.
Everything in quotes in the schema is a relationship.
All entities before the relationship are valid heads.
All entities after -> are valid tails.
In this schema text, "rdf:type" between class names means class hierarchy/type compatibility for reasoning. Do not treat class-to-class "rdf:type" as a literal SPARQL triple.
If A -> rdf:type -> B, then A is a child/specialization of B.
When the question asks about entities of type B, also consider children of B as valid subclasses when appropriate.

Return exactly this format:
Reasoning path:
<one or more uninterrupted chains using arrows>

Interpretation:
<brief interpretation>

{schema_paths}

Question:
{question}
"""

    def __init__(
        self,
        llm,
        schema_paths_path: str,
        linkml_schema_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
    ):
        self.llm = llm
        self.schema_paths_path = Path(schema_paths_path)
        self.linkml_schema_path = Path(linkml_schema_path) if linkml_schema_path else None
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.user_prompt_template = user_prompt_template or self.DEFAULT_USER_PROMPT_TEMPLATE
        self._schema_paths_cache: Optional[str] = None
        self._class_uri_cache: Optional[dict[str, str]] = None
        self._context_cache: dict[str, str] = {}
        self._artifact_cache: dict[str, dict[str, str]] = {}
        self._last_artifacts: dict[str, str] = {}

    def get_context(self, question: str, question_id: Optional[str] = None) -> str:
        cache_key = f"{question_id or ''}::{question}"
        if cache_key in self._context_cache:
            self._last_artifacts = self._artifact_cache.get(cache_key, {})
            return self._context_cache[cache_key]

        prompt = self._build_prompt(question)
       # print(f"Generated reasoning path prompt:\n{prompt}\n")
        response = self.llm(prompt)
        reasoning_path = (response or "").strip()
        #resolved_reasoning_path = self.resolve_class_names(reasoning_path)
        #class_uri_map = self._load_class_uri_map()
        #context = self.format_context(reasoning_path, resolved_reasoning_path)
        artifacts = {
            "reasoning_path_schema_paths_path": str(self.schema_paths_path),
            "reasoning_path_linkml_schema_path": str(self.linkml_schema_path or ""),
            "reasoning_path_schema": self._load_schema_paths(),
            "reasoning_path_prompt": prompt,
            "reasoning_path_generated": reasoning_path,
            #"reasoning_path_resolved": resolved_reasoning_path,
           # "reasoning_path_class_uri_map": json.dumps(class_uri_map, ensure_ascii=False, sort_keys=True),
            #"reasoning_path_context": context,
        }
        #self._context_cache[cache_key] = context
        self._artifact_cache[cache_key] = artifacts
        self._last_artifacts = artifacts
        return reasoning_path

    def get_last_artifacts(self) -> dict[str, str]:
        return dict(self._last_artifacts)

    def _build_prompt(self, question: str) -> str:
        user_prompt = (
            self.user_prompt_template
            .replace("{schema_paths}", self._load_schema_paths())
            .replace("{question}", question)
        )
        return f"{self.system_prompt}\n\n{user_prompt}"

    def _load_schema_paths(self) -> str:
        if self._schema_paths_cache is None:
            if not self.schema_paths_path.exists():
                raise FileNotFoundError(f"Schema paths file not found: {self.schema_paths_path}")
            self._schema_paths_cache = self.schema_paths_path.read_text(encoding="utf-8")
        return self._schema_paths_cache

    def _load_class_uri_map(self) -> dict[str, str]:
        if self._class_uri_cache is not None:
            return self._class_uri_cache

        self._class_uri_cache = {}
        if not self.linkml_schema_path:
            return self._class_uri_cache
        if not self.linkml_schema_path.exists():
            raise FileNotFoundError(f"LinkML schema file not found: {self.linkml_schema_path}")

        schema = yaml.safe_load(self.linkml_schema_path.read_text(encoding="utf-8")) or {}
        classes = schema.get("classes", {})
        if not isinstance(classes, dict):
            return self._class_uri_cache

        for class_name, class_def in classes.items():
            if not isinstance(class_name, str) or not isinstance(class_def, dict):
                continue
            class_uri = class_def.get("class_uri")
            if isinstance(class_uri, str) and class_uri.strip():
                self._class_uri_cache[class_name] = class_uri.strip()

        return self._class_uri_cache

    def resolve_class_names(self, reasoning_path: str) -> str:
        class_uri_map = self._load_class_uri_map()
        resolved_path = reasoning_path

        for class_name, class_uri in sorted(class_uri_map.items(), key=lambda item: len(item[0]), reverse=True):
            pattern = re.compile(
                rf"(?<![:\w\[]){re.escape(class_name)}(?!\s*\[)(?![\w\]])"
            )
            resolved_path = pattern.sub(class_uri, resolved_path)

        return resolved_path

    @staticmethod
    def format_context(reasoning_path: str, resolved_reasoning_path: str) -> str:
        resolved_body = resolved_reasoning_path.strip()
        if resolved_body.startswith("Reasoning path:"):
            resolved_body = resolved_body[len("Reasoning path:"):].lstrip()
        return "\n".join([
            "Reasoning path:",
            resolved_body,
            "",
            "Class-to-class rdf:type in the reasoning path indicates hierarchy/type compatibility, not a literal SPARQL triple.",
            "Use only IRIs and prefixes that are justified by the reasoning path and schema context.",
        ])
