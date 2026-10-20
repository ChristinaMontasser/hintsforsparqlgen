# main.py
import hydra
import time
import importlib
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.ner_context_builder import NERSchemaContextBuilder
from src.schema_loader import OntologyVectorStore, SchemaLoader, SchemaMode
from src.schema_path_builder import SchemaPathContextBuilder
from src.sparql_chain import SPARQLGenerator
from src.reasoning_path_builder import ReasoningPathBuilder
from src.validate_query import QueryValidator, ResultSerializer, SyntaxValidator

def instantiate_llm_backend(cfg):
    # print(f"Instantiating LLM backend from config: {cfg.llm}")
    model_name, class_name = cfg.llm.rsplit(".", 1)
    module = importlib.import_module(model_name)
    cls = getattr(module, class_name)
    # print(f"Class obtained: {cls}")
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
        k=vs_cfg.rag_top_k
    )
    
    # Build index if it doesn't exist, otherwise load
    if not os.path.exists(vs_cfg.index_path):
        print(f"Building vector index from {cfg.schema.ontology_path}...")
        vector_store.build_and_save(cfg.schema.ontology_path)
    else:
        print(f"Loading vector index from {vs_cfg.index_path}")
        vector_store.load()
    
    return vector_store


def get_schema_label(mode: SchemaMode) -> str:
    """Return a clean label for output column naming."""
    labels = {
        SchemaMode.FULL_SCHEMA: "With_FullSchema",
        SchemaMode.SMALL_SCHEMA: "Only_ClassesProperties", 
        SchemaMode.RAG_TRIPLES: "With_RAG",
        SchemaMode.NER: "With_NER",
        SchemaMode.SCHEMA_PATH: "With_SchemaPath",
        SchemaMode.REASONING_PATH: "With_ReasoningPath",
        SchemaMode.LEXICON: "With_Lexicon",
        SchemaMode.ZERO_SHOT: "No_Schema"
    }
    return labels.get(mode, "Unknown_Schema")


def sanitize_filename_part(value: str) -> str:
    value = str(value or "unknown")
    value = value.replace("/", "_").replace("\\", "_")
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return value or "unknown"


def infer_graph_name(system_message: str) -> str:
    match = re.search(r"GRAPH\s*<hydroturtle/([^>]+)>", system_message or "")
    if match:
        return sanitize_filename_part(match.group(1))
    return "unknown_graph"


def infer_reasoning_name(cfg: DictConfig, schema_mode: SchemaMode) -> str:
    if schema_mode == SchemaMode.REASONING_PATH:
        return sanitize_filename_part(Path(cfg.graph.full_schema_path).stem)
    if schema_mode == SchemaMode.SCHEMA_PATH:
        return "schema_path"
    if schema_mode == SchemaMode.NER:
        return "ner"
    return schema_mode.value


def get_selected_prompt_hints(cfg: DictConfig) -> list[str]:
    hints = OmegaConf.select(cfg, "prompt_hints", default=[])
    if not hints:
        return []
    if isinstance(hints, str):
        return [hints]
    return [str(hint) for hint in hints if str(hint).strip()]

# This function composes the system message by combining the base system message with any selected hint blocks.
def compose_system_message(cfg: DictConfig) -> str:
    base_message = str(cfg.prompt.system_message).strip()
    hint_blocks = OmegaConf.select(cfg, "prompt.tag_blocks", default=None)
    if hint_blocks is None:
        hint_blocks = OmegaConf.select(cfg, "prompt.hint_blocks", default={}) or {}
    selected_hints = get_selected_prompt_hints(cfg)

    parts = [base_message]
    for hint_name in selected_hints:
        hint_block = hint_blocks.get(hint_name)
        if hint_block:
            parts.append(str(hint_block).strip())
    return "\n\n".join(part for part in parts if part)


def build_dynamic_output_path(cfg: DictConfig, model_name: str, schema_mode: SchemaMode, system_message: str | None = None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    dataset_name = sanitize_filename_part(Path(cfg.graph.questions_path).stem)
    graph_name = cfg.graph.name  # Default graph name
    #graph_name = #Todo: given as a parameter
    llm_name = sanitize_filename_part(model_name)
    experiment_name = sanitize_filename_part(OmegaConf.select(cfg, "experiment_name", default="default"))
    reasoning_name = infer_reasoning_name(cfg, schema_mode)
    return f"{timestamp}_{dataset_name}_{graph_name}_{llm_name}_{experiment_name}_{reasoning_name}_tag_prompt_system_user.csv"


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    output_dir = Path(cfg.output_dir)
    #Make sure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_cfg = OmegaConf.select(cfg, "experiment", default=None)
    if experiment_cfg is not None:
        cfg = OmegaConf.merge(cfg, experiment_cfg)
       # print(cfg.pretty())
    system_message = compose_system_message(cfg)
   # print(OmegaConf.to_yaml(cfg))
    # --- Setup sparql query validator ---
    # Prepare metadata for this run
    # --- Setup Schema Loader ---
    schema_mode = SchemaMode(cfg.schema.mode)
    print(f"Schema Mode: {schema_mode.value}")
    
    # Initialize vector store if needed (only once, before loop)
    vector_store = None
    ner_builder = None
    schema_path_builder = None
    reasoning_path_builder = None
    if schema_mode == SchemaMode.RAG_TRIPLES:
        vector_store = initialize_vector_store(cfg)
    
    print(" Instantiating LLM backend...")
    llm_cfg = cfg.llm
    llm = instantiate_llm_backend(llm_cfg)
    model_name = llm_cfg.model_name.replace("/", "_")  # TODO: Sanitize for filenames
    print(f"LLM: {model_name}")

    evaluation_llm_cfg = cfg.evaluation_llm
    evaluation_enabled = bool(OmegaConf.select(cfg, "evaluation.enabled", default=True))
    evaluation_llm = (
        instantiate_llm_backend(evaluation_llm_cfg)
        if evaluation_enabled
        else None
    )
    if evaluation_enabled:
        print(f"Evaluation LLM: {evaluation_llm_cfg.model_name}")
    else:
        print("Evaluation pipeline disabled")

    use_reasoning_path_hint = bool(OmegaConf.select(cfg, "use_reasoning_path_hint", default=False))

    query_validator = None
    if evaluation_enabled:
        query_validator = QueryValidator(
            cfg.graph.validator_api,
            llm=evaluation_llm,
            llm_judge_prompt=(
                "You should assess whether or not the following two results from the bindings are equivalent. "
                "The question is: {question}\n\n"
                "Return JSON only with exactly two keys: 'decision' and 'reason'.\n"
                "'decision' must be either 'Yes' or 'No'.\n"
                "'reason' must explain why the two bindings are equivalent or not equivalent.\n\n"
                "Gold result bindings:\n{gold_results}\n\n"
                "Generated result bindings:\n{generated_results}"
            ),
        )

    if schema_mode == SchemaMode.NER:
        print(" Instantiating NER backend...")
        ner_llm_cfg = get_mode_llm_cfg(cfg, "ner_llm")
        ner_llm = instantiate_llm_backend(ner_llm_cfg)
        ner_builder = NERSchemaContextBuilder(
            llm=ner_llm,
            schema_txt=cfg.graph.small_schema_path,
            system_prompt=cfg.ner.system_prompt,
            user_prompt_template=cfg.ner.user_prompt_template,
        )

    if schema_mode == SchemaMode.SCHEMA_PATH:
        print(" Instantiating schema-path backend...")
        schema_path_llm_cfg = get_mode_llm_cfg(cfg, "schema_path_llm")
        schema_path_llm = instantiate_llm_backend(schema_path_llm_cfg)
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

    if schema_mode == SchemaMode.REASONING_PATH or use_reasoning_path_hint:
        print(" Instantiating reasoning-path backend...")
        reasoning_path_llm_cfg = get_mode_llm_cfg(cfg, "reasoning_path_llm")
        reasoning_path_llm = instantiate_llm_backend(reasoning_path_llm_cfg)
        reasoning_path_builder = ReasoningPathBuilder(
            llm=reasoning_path_llm,
            schema_paths_path=cfg.graph.full_schema_path,
            #linkml_schema_path=cfg.graph.reasoning_path.get("linkml_schema_path"),
            system_prompt=cfg.reasoning_path.system_prompt,
            user_prompt_template=cfg.reasoning_path.user_prompt_template,
        )

    # Prepare schema loader config
    loader_config = {
        'full_schema_path': cfg.graph.full_schema_path,
        'small_schema_path': cfg.graph.small_schema_path,
        'lexicon_schema_path': cfg.graph.lexicon_schema_path,
        'vector_store': vector_store,
        'ner_builder': ner_builder,
        'schema_path_builder': schema_path_builder,
        'reasoning_path_builder': reasoning_path_builder,
    }
    schema_loader = SchemaLoader(schema_mode, loader_config)
    schema_label = get_schema_label(schema_mode)
    
    # ---  Load Data ---
    questions = pd.read_csv(cfg.graph.questions_path, sep=";")
    max_q = cfg.data.max_questions
    if max_q > 0:
        questions = questions.head(max_q)
        print(f" Processing first {max_q} questions (test mode)")
    
    generator = SPARQLGenerator(llm, system_message)
    
    # Prepare output column name
    output_col = f'{model_name}_{schema_label}_generated_sparql'
    output_path = build_dynamic_output_path(
        cfg, model_name, schema_mode, system_message=system_message
    )
    output_file = output_dir / output_path
    base_path, _ = os.path.splitext(output_path)
    excel_output_path = output_dir / f"{base_path}.xlsx"
    artifacts_output_path = output_dir / f"{base_path}_schema_paths.csv"
    results = []
    path_artifacts = []
    # --- 3. Main Processing Loop ---
    for index, row in questions.iterrows():    
        metadata = {
            "model_name": model_name,
            "experiment_name": OmegaConf.select(cfg, "experiment_name", default="default"),
            "schema_mode": schema_mode.value,
            "schema_label": schema_label,
            "question_id": index,
            "run_timestamp": pd.Timestamp.now().isoformat(),
        }
        question = row['Hand-Written Question']
        actual_sparql = "" if pd.isna(row["Query"]) else str(row["Query"]).strip()
        print(f"\n[{index + 1}/{len(questions)}] Q: {question[:80]}{'...' if len(question)>80 else ''}")
        artifact_count_before = len(path_artifacts)
        
        # Query-aware modes (RAG / NER) use the question to build context.
        context = schema_loader.get_context(query=question, question_id=f"Q{index + 1}")
        if use_reasoning_path_hint and schema_mode != SchemaMode.REASONING_PATH and reasoning_path_builder:
            reasoning_context = reasoning_path_builder.get_context(question, question_id=f"Q{index + 1}")
            context = "\n\n".join(
                part for part in [str(context), "Reasoning path hint:", reasoning_context] if part
            )
        if schema_mode == SchemaMode.SCHEMA_PATH and schema_path_builder:
            artifacts = schema_path_builder.get_last_artifacts()
            metadata.update(artifacts)
            path_artifacts.append({
                "question": question,
                "question_id": index,
                "run_question_id": f"Q{index + 1}",
                "model_name": model_name,
                "experiment_name": OmegaConf.select(cfg, "experiment_name", default="default"),
                "schema_mode": schema_mode.value,
                "schema_label": schema_label,
                "run_timestamp": metadata["run_timestamp"],
                **artifacts,
            })
        if (schema_mode == SchemaMode.REASONING_PATH or use_reasoning_path_hint) and reasoning_path_builder:
            artifacts = reasoning_path_builder.get_last_artifacts()
            metadata.update(artifacts)
            path_artifacts.append({
                "question": question,
                "question_id": index,
                "run_question_id": f"Q{index + 1}",
                "model_name": model_name,
                "experiment_name": OmegaConf.select(cfg, "experiment_name", default="default"),
                "schema_mode": schema_mode.value,
                "schema_label": schema_label,
                "run_timestamp": metadata["run_timestamp"],
                **artifacts,
            })
        print(f"Full schema path: {cfg.graph.full_schema_path}")
        #print(context)
        # 1. Record start time as numeric value BEFORE generation
        start_time = time.time()
        # Generate SPARQL
        generated_sparql = generator.generate(
            user_message=cfg.prompt.user_message,
            question=question,
            schema=context,  # Pass the retrieved/loaded schema context
        )
        execution_time = time.time() - start_time
        print(f" Generated: {generated_sparql[:100]}{'...' if len(generated_sparql)>100 else ''}")

        if query_validator is None:
            sparql_similarities = {
                "structure_similarity": {},
                "content_similarity": None,
                "language_similarity": {},
            }
            syntax_validation_result = SyntaxValidator().validate(generated_sparql)
            results_similarities = {
                "query_type": "skipped",
                "error": "Evaluation pipeline disabled",
            }
            is_valid = syntax_validation_result.grammar_ok
        else:
            (
                sparql_similarities,
                syntax_validation_result,
                results_similarities,
                is_valid,
            ) = query_validator.process_queries(
                generated_sparql, actual_sparql, question
            )

        # Serialize to flat row
         # Serialize to flat row
        result_row = ResultSerializer.serialize_row(
            question=question,
            generated_sparql=generated_sparql,
            actual_sparql= actual_sparql,
            sparql_similarity=sparql_similarities,
            validation_result=syntax_validation_result,
            execution_time=execution_time,
            is_valid_final=is_valid,
            result_similarity=results_similarities,
            metadata=metadata        )
        #print(result_row)

        results.append(result_row)
        pd.DataFrame([result_row]).to_csv(
            output_file,
            mode="a",
            header=not output_file.exists(),
            index=False,
        )

        if len(path_artifacts) > artifact_count_before:
            artifact_row = path_artifacts[-1]
            pd.DataFrame([artifact_row]).to_csv(
                artifacts_output_path,
                mode="a",
                header=not artifacts_output_path.exists(),
                index=False,
            )
        print(f" Saved progress to {output_file}")
        
        # Store result
        #questions.loc[index, output_col] = sparql
        
        #  early break 
        # if index >= 2: break

    # Save Results ---
    results_df = pd.DataFrame(results)
    results_df.to_excel(excel_output_path, index=False, engine="openpyxl")

    if path_artifacts:
        print(f" Path artifacts saved to {artifacts_output_path}")
   # results_df.to_csv(output_path, index=False)
    # questions.to_csv(output_path, index=False)
    print(f"\n Results saved to {output_file}")
    print(f" Excel results saved to {excel_output_path}")
    print(f" Output column: '{output_col}'")


if __name__ == "__main__":
    main()
