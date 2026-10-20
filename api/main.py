# api/main.py

import importlib
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import requests
from hydra import compose
from fastapi import FastAPI, HTTPException
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel

from src.ner_context_builder import NERSchemaContextBuilder
from src.schema_loader import (
    OntologyVectorStore,
    SchemaLoader,
    SchemaMode,
)
from src.schema_path_builder import SchemaPathContextBuilder
from src.sparql_chain import SPARQLGenerator
from src.reasoning_path_builder import ReasoningPathBuilder
from api.models import GenerateRequest, GenerateResponse
import uvicorn

# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="SPARQL Generation API",
    description="Generate SPARQL queries from natural-language questions.",
    version="1.0.0",
)

AVAILABLE_LLM_CONFIGS = frozenset(
    path.stem
    for path in (
        Path(__file__).resolve().parents[1] / "config" / "llm"
    ).glob("*.yaml")
)


# ============================================================
# Helper functions
# ============================================================

def instantiate_llm_backend(cfg):
    model_name, class_name = cfg.llm.rsplit(".", 1)
    module = importlib.import_module(model_name)
    cls = getattr(module, class_name)
    return cls(cfg)


def get_mode_llm_cfg(cfg: DictConfig, key: str):
    selected = OmegaConf.select(cfg, key, default=None)

    if selected is None:
        return cfg.llm

    return selected


def initialize_vector_store(cfg: DictConfig) -> OntologyVectorStore:
    """Initialize or load the vector store for RAG mode."""

    vs_cfg = cfg.schema.vector_store

    vector_store = OntologyVectorStore(
        index_path=vs_cfg.index_path,
        chunks_path=vs_cfg.chunks_path,
        model_name=vs_cfg.embedding_model,
        k=vs_cfg.rag_top_k,
    )

    if not Path(vs_cfg.index_path).exists():
        print(
            f"Building vector index from "
            f"{cfg.graph.ontology_path}..."
        )

        vector_store.build_and_save(
            cfg.graph.ontology_path
        )

    else:
        print(
            f"Loading vector index from "
            f"{vs_cfg.index_path}"
        )

        vector_store.load()

    return vector_store


def get_selected_prompt_hints(cfg: DictConfig) -> list[str]:
    hints = OmegaConf.select(
        cfg,
        "prompt_hints",
        default=[],
    )

    if not hints:
        return []

    if isinstance(hints, str):
        return [hints]

    return [
        str(hint)
        for hint in hints
        if str(hint).strip()
    ]


def compose_system_message(cfg: DictConfig) -> str:
    base_message = str(
        cfg.prompt.system_message
    ).strip()

    hint_blocks = OmegaConf.select(
        cfg,
        "prompt.tag_blocks",
        default=None,
    )

    if hint_blocks is None:
        hint_blocks = OmegaConf.select(
            cfg,
            "prompt.hint_blocks",
            default={},
        ) or {}

    selected_hints = get_selected_prompt_hints(cfg)

    include_all_hints = bool(
        OmegaConf.select(
            cfg,
            "include_all_prompt_hints",
            default=False,
        )
    )

    if include_all_hints:
        selected_hints = list(hint_blocks.keys())

    parts = [base_message]

    for hint_name in selected_hints:
        hint_block = hint_blocks.get(hint_name)

        if hint_block:
            parts.append(
                str(hint_block).strip()
            )

    return "\n\n".join(
        part
        for part in parts
        if part
    )


def execute_sparql_query(
    query: str,
    endpoint: str,
) -> dict:
    headers = {
        "Content-Type": "application/sparql-query",
        "Accept": "application/sparql-results+json",
    }

    try:
        response = requests.post(
            endpoint,
            data=query,
            headers=headers,
            timeout=90,
        )
        response.raise_for_status()
        return {
            "success": True,
            "data": response.json(),
            "error": None,
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "data": None,
            "error": "TIMEOUT",
        }
    except requests.exceptions.RequestException as exc:
        return {
            "success": False,
            "data": None,
            "error": f"HTTP_ERROR: {exc}",
        }
    except ValueError as exc:
        return {
            "success": False,
            "data": None,
            "error": f"INVALID_JSON_RESPONSE: {exc}",
        }


def clean_sparql_query(query: str) -> str:
    cleaned_query = query.strip()
    cleaned_query = re.sub(
        r"^```(?:sparql)?\s*\n?",
        "",
        cleaned_query,
        flags=re.IGNORECASE,
    )
    cleaned_query = re.sub(
        r"\n?```\s*$",
        "",
        cleaned_query,
    )
    return cleaned_query.strip()


def compose_runtime_config(
    graph_name: str,
    schema_mode: str,
    llm_name: str | None = None,
) -> DictConfig:
    """Compose graph-specific runtime configuration through Hydra."""

    config_dir = Path(__file__).resolve().parents[1] / "config"

    required_configs = (
        config_dir / "graphs" / f"{graph_name}.yaml",
        config_dir / "prompt" / f"{graph_name}.yaml",
        config_dir / "reasoning_path_prompt" / f"{graph_name}.yaml",
    )

    if not all(path.is_file() for path in required_configs):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported graph '{graph_name}'.",
        )

    overrides = [
        f"graphs@_global_={graph_name}",
        f"prompt={graph_name}",
        f"reasoning_path_prompt@_global_={graph_name}",
        f"schema.mode={schema_mode}",
    ]

    if llm_name is not None:
        if llm_name not in AVAILABLE_LLM_CONFIGS:
            available = ", ".join(sorted(AVAILABLE_LLM_CONFIGS))
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported LLM '{llm_name}'. "
                    f"Available LLMs: {available}."
                ),
            )

        overrides.extend(
            [
                f"llm={llm_name}",
                f"llm@ner_llm={llm_name}",
                f"llm@reasoning_path_llm={llm_name}",
            ]
        )

    return compose(
        config_name="config",
        overrides=overrides,
    )


# ============================================================
# API application state
# ============================================================

class ApplicationState:
    """
    Holds all expensive components that should be initialized
    only once when the application starts.
    """

    cfg: DictConfig | None = None

    llm = None
    generator = None

    schema_loader = None

    reasoning_path_builder = None
    schema_path_builder = None
    ner_builder = None

    schema_mode = None
    graph_name = None
    llm_name = None


state = ApplicationState()


# ============================================================
# Application initialization
# ============================================================

def initialize_application(
    cfg: DictConfig,
    llm_name: str | None = None,
):
    """
    Initialize the LLM, schema loader, builders and generator.
    """

    state.cfg = cfg
    state.llm_name = llm_name

    # --------------------------------------------------------
    # Schema mode
    # --------------------------------------------------------

    schema_mode = SchemaMode(cfg.schema.mode)

    state.schema_mode = schema_mode

    print(
        f"Schema Mode: {schema_mode.value}"
    )

    # --------------------------------------------------------
    # Vector store
    # --------------------------------------------------------

    vector_store = None

    if schema_mode == SchemaMode.RAG_TRIPLES:
        vector_store = initialize_vector_store(cfg)

    # --------------------------------------------------------
    # Main LLM
    # --------------------------------------------------------

    print("Instantiating LLM backend...")

    llm = instantiate_llm_backend(
        cfg.llm
    )

    state.llm = llm

    print(
        f"LLM: {cfg.llm.model_name}"
    )

    # --------------------------------------------------------
    # Optional NER backend
    # --------------------------------------------------------

    ner_builder = None

    if schema_mode == SchemaMode.NER:

        print(
            "Instantiating NER backend..."
        )

        ner_llm_cfg = get_mode_llm_cfg(
            cfg,
            "ner_llm",
        )

        ner_llm = instantiate_llm_backend(
            ner_llm_cfg
        )

        ner_builder = NERSchemaContextBuilder(
            llm=ner_llm,
            schema_txt=cfg.graph.small_schema_path,
            system_prompt=cfg.ner.system_prompt,
            user_prompt_template=cfg.ner.user_prompt_template,
        )

        state.ner_builder = ner_builder

    # --------------------------------------------------------
    # Schema-path backend
    # --------------------------------------------------------

    schema_path_builder = None

    if schema_mode == SchemaMode.SCHEMA_PATH:

        print(
            "Instantiating schema-path backend..."
        )

        schema_path_llm_cfg = get_mode_llm_cfg(
            cfg,
            "schema_path_llm",
        )

        schema_path_llm = instantiate_llm_backend(
            schema_path_llm_cfg
        )

        schema_path_builder = SchemaPathContextBuilder(
            llm=schema_path_llm,
            lexicons_path=cfg.schema_path.lexicons_path,
            fuzzy_threshold=cfg.schema_path.fuzzy_threshold,
            max_candidate_classes=cfg.schema_path.max_candidate_classes,
            max_candidate_relations=cfg.schema_path.max_candidate_relations,
            use_full_schema_paths=cfg.schema_path.use_full_schema_paths,
            schema_path_scope=cfg.schema_path.schema_path_scope,
            system_prompt=cfg.schema_path.system_prompt,
            user_prompt_template=cfg.schema_path.user_prompt_template,
        )

        state.schema_path_builder = (
            schema_path_builder
        )

    # --------------------------------------------------------
    # Reasoning-path backend
    # --------------------------------------------------------

    use_reasoning_path_hint = bool(
        OmegaConf.select(
            cfg,
            "use_reasoning_path_hint",
            default=False,
        )
    )

    reasoning_path_builder = None

    if (
        schema_mode == SchemaMode.REASONING_PATH
        or use_reasoning_path_hint
    ):

        print(
            "Instantiating reasoning-path backend..."
        )

        reasoning_path_llm_cfg = get_mode_llm_cfg(
            cfg,
            "reasoning_path_llm",
        )

        reasoning_path_llm = instantiate_llm_backend(
            reasoning_path_llm_cfg
        )

        reasoning_path_builder = ReasoningPathBuilder(
            llm=reasoning_path_llm,
            schema_paths_path=cfg.graph.full_schema_path,
            system_prompt=cfg.reasoning_path.system_prompt,
            user_prompt_template=cfg.reasoning_path.user_prompt_template,
        )

        state.reasoning_path_builder = (
            reasoning_path_builder
        )

    # --------------------------------------------------------
    # Schema loader
    # --------------------------------------------------------

    loader_config = {
        "full_schema_path": cfg.graph.full_schema_path,
        "small_schema_path": cfg.graph.small_schema_path,
        "lexicon_schema_path": cfg.graph.lexicon_schema_path,
        "vector_store": vector_store,
        "ner_builder": ner_builder,
        "schema_path_builder": schema_path_builder,
        "reasoning_path_builder": reasoning_path_builder,
    }

    state.schema_loader = SchemaLoader(
        schema_mode,
        loader_config,
    )

    # --------------------------------------------------------
    # System prompt
    # --------------------------------------------------------

    system_message = compose_system_message(
        cfg
    )

    # --------------------------------------------------------
    # SPARQL generator
    # --------------------------------------------------------

    state.generator = SPARQLGenerator(
        llm,
        system_message,
    )

    state.graph_name = str(
        cfg.graph.name
    )

    print(
        "Application initialization complete."
    )


# ============================================================
# Generate endpoint
# ============================================================

@app.post(
    "/generate",
    response_model=GenerateResponse,
)
def generate(request: GenerateRequest):

    if state.generator is None:
        raise HTTPException(
            status_code=500,
            detail="Application has not been initialized.",
        )

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    # --------------------------------------------------------
    # Graph
    # --------------------------------------------------------

    graph_name = (
        request.graph.strip()
        if request.graph
        else state.graph_name
    )

    # --------------------------------------------------------
    # Schema mode
    # --------------------------------------------------------

    requested_schema_mode = (
        request.schema_mode.strip()
        if request.schema_mode
        else state.schema_mode.value
    )

    try:
        requested_schema_mode = SchemaMode(
            requested_schema_mode
        ).value
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported schema mode '{requested_schema_mode}'.",
        ) from exc

    requested_llm = (
        request.llm.strip()
        if request.llm
        else state.llm_name
    )

    if requested_llm is not None and requested_llm not in AVAILABLE_LLM_CONFIGS:
        available = ", ".join(sorted(AVAILABLE_LLM_CONFIGS))
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported LLM '{requested_llm}'. "
                f"Available LLMs: {available}."
            ),
        )

    if (
        graph_name != state.graph_name
        or requested_schema_mode != state.schema_mode.value
        or requested_llm != state.llm_name
    ):
        runtime_cfg = compose_runtime_config(
            graph_name,
            requested_schema_mode,
            requested_llm,
        )
        initialize_application(
            runtime_cfg,
            llm_name=requested_llm,
        )
    #print(state.cfg)

    # --------------------------------------------------------
    # Build schema context
    # --------------------------------------------------------

    start_time = time.time()

    context = state.schema_loader.get_context(
        query=question,
        question_id="API_REQUEST",
    )

    # --------------------------------------------------------
    # Optional reasoning path
    # --------------------------------------------------------

    use_reasoning_path_hint = bool(
        OmegaConf.select(
            state.cfg,
            "use_reasoning_path_hint",
            default=False,
        )
    )

    if (
        use_reasoning_path_hint
        and state.schema_mode != SchemaMode.REASONING_PATH
        and state.reasoning_path_builder
    ):

        reasoning_context = (
            state.reasoning_path_builder.get_context(
                question,
                question_id="API_REQUEST",
            )
        )

        context = "\n\n".join(
            part
            for part in [
                str(context),
                "Reasoning path hint:",
                reasoning_context,
            ]
            if part
        )

    # --------------------------------------------------------
    # Generate SPARQL
    # --------------------------------------------------------
    #print(context)
    generated_sparql = state.generator.generate(
        user_message=state.cfg.prompt.user_message,
        question=question,
        schema=context,
    )
    generated_sparql = clean_sparql_query(generated_sparql)

    query_result = None
    query_execution_time = None

    if request.execute:
        query_execution_start = time.time()
        query_result = execute_sparql_query(
            generated_sparql,
            str(state.cfg.graph.validator_api),
        )
        query_execution_time = time.time() - query_execution_start

    execution_time = (
        time.time() - start_time
    )

    print(
        f"Question: {question}"
    )

    print(
        f"Generated SPARQL: "
        f"{generated_sparql}"
    )

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return GenerateResponse(
        question=question,
        graph=graph_name,
        schema_mode=requested_schema_mode,
        generated_sparql=generated_sparql,
        execution_time=execution_time,
        query_result=query_result,
        query_execution_time=query_execution_time,
    )


# ============================================================
# Hydra entry point
# ============================================================

@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="config",
)
def main(cfg: DictConfig):

    initialize_application(cfg)
    print(
        "Starting FastAPI server on "
        "http://0.0.0.0:8000"
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )


if __name__ == "__main__":
    main()
