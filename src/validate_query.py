import requests
import json
import re
from src.similarity_measure import  Structure_Similarity, Language_Similarity, Content_Similarity
import pandas as pd 
from rdflib.plugins.sparql.parser import parseQuery
from rdflib.plugins.sparql.processor import prepareQuery

def validate_query(query, endpoint="https://sparql.knowledgehub.nfdi4earth.de"):
    headers = {
            "Content-Type": "application/sparql-query",
            "Accept": "application/sparql-results+json"
        }
        
    try:
        response = requests.post(endpoint, data=query, headers=headers, timeout=90)
        response.raise_for_status()
        
        return {
            "success": True,
            "data": response.json(),
            "error": None
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "data": None,
            "error": "TIMEOUT"
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "data": None,
            "error": f"HTTP_ERROR: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "data": None,
            "error": f"UNEXPECTED: {str(e)}"
        }
    



import requests
from collections import defaultdict
import json

class QueryValidationError(Exception):
    """Custom exception for query validation errors."""
    pass

# Input Generated Query, actual Query 
# Parameters: Endpoint Sparql API, Santax Validtor [could be done through the api itself], 
# Output: for each question, Santex correct. Answer Correct. Answer Similarity 

# Syntax Correct --> Either through Sparql API or Sparql checker. [to be defined, what and why]
# Answer Correct --> is the result correct, Doe we retrieve the same data we want if we run the other query
# To be defined (retrieved elements are related to the question and valid on the schema)
# Answer Similarity --> Query Similarity, from writing perceptive. 

from dataclasses import dataclass
from rdflib.plugins.sparql.parser import parseQuery
from rdflib.plugins.sparql.processor import prepareQuery


@dataclass
class ValidationResult:
    grammar_ok: bool
    grammar_error: str | None
    preparation_ok: bool
    preparation_error: str | None


class SyntaxValidator:

    def validate(self, query: str):
        grammar_ok = False
        preparation_ok = False
        grammar_error = None
        preparation_error = None

        # Grammar validation
        try:
            parseQuery(query)
            grammar_ok = True
        except Exception as e:
            grammar_error = str(e)
            return ValidationResult(
                grammar_ok=False,
                grammar_error=grammar_error,
                preparation_ok=False,
                preparation_error=None
            )

        # Preparation validation (only if grammar passes)
        try:
            prepareQuery(query)
            preparation_ok = True
        except Exception as e:
            preparation_error = str(e)

        return ValidationResult(
            grammar_ok=grammar_ok,
            grammar_error=None,
            preparation_ok=preparation_ok,
            preparation_error=preparation_error
        )

class QuerySimilarity:
    def __init__(self, ):
        self.structure_similarity = Structure_Similarity()
        self.language_similarity = Language_Similarity()
        self.content_similarity = Content_Similarity()
        self.similarities = {
            'structure_similarity': 0,
            'content_similarity': 0,
            'language_similarity': 0
        }
    def measure_similarities(self, generated_query: str, actual_query: str)-> float:
        #Vector Embedding or Lexical Analysis 
        ##Based on the retrieved docs, contained. , the similarity is defined based on the set of data retrieve
        self.similarities['structure_similarity'] = self.structure_similarity.measure_similarity(generated_query, actual_query)
        self.similarities['content_similarity'] = self.content_similarity.measure_similarity(generated_query, actual_query)
        self.similarities['language_similarity'] = self.language_similarity.measure_similarity(generated_query, actual_query)
 
        return self.similarities


class LLMResultEquivalenceJudge:
    """Small LLM judge that compares gold and generated SPARQL result bindings."""

    DEFAULT_PROMPT = (
        "You should assess whether or not the following two results from the bindings are equivalent. "
        "The question is: {question}\n\n"
        "Return JSON only with exactly two keys: 'decision' and 'reason'.\n"
        "'decision' must be either 'Yes' or 'No'.\n"
        "'reason' must explain why the two bindings are equivalent or not equivalent.\n\n"
        "Gold result bindings:\n{gold_results}\n\n"
        "Generated result bindings:\n{generated_results}"
    )

    def __init__(self, llm=None, prompt_template: str | None = None):
        self.llm = llm
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT

    def judge(self, question: str, gold_results: dict, generated_results: dict) -> dict:
        if self.llm is None:
            return {
                "decision": "No",
                "reason": "LLM judge unavailable: no backend configured.",
            }

        prompt = self.prompt_template.format(
            question=question,
            gold_results=json.dumps(gold_results, ensure_ascii=False, indent=2),
            generated_results=json.dumps(generated_results, ensure_ascii=False, indent=2),
        )

        try:
            raw_response = self.llm(prompt)
            parsed = self._parse_response(raw_response)
            if parsed:
                return parsed
        except Exception as exc:
            return {
                "decision": "No",
                "reason": f"LLM judge failed: {exc}",
            }

        return {
            "decision": "No",
            "reason": "LLM judge did not return valid JSON.",
        }

    @staticmethod
    def _parse_response(raw_response: str) -> dict | None:
        text = str(raw_response or "").strip()
        if not text:
            return None

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            payload = match.group(0)
        else:
            payload = text

        try:
            parsed = json.loads(payload)
            if isinstance(parsed, dict) and "decision" in parsed and "reason" in parsed:
                decision = str(parsed["decision"]).strip().capitalize()
                if decision not in {"Yes", "No"}:
                    decision = "No"
                return {
                    "decision": decision,
                    "reason": str(parsed.get("reason") or "LLM judge did not provide a reason."),
                }
        except Exception:
            return None

        return None


class ResultSimilarity:
    def __init__(
        self,
        endpoint: str="https://sparql.knowledgehub.nfdi4earth.de",
        llm_judge: LLMResultEquivalenceJudge | None = None,
        question: str = "",
    ):
        self.endpoint = endpoint
        self.llm_judge = llm_judge
        self.question = question

    def post_query(self, query: str) -> dict:
        """
        Send query to SPARQL endpoint. 
        Always returns a dict with 'success' flag and 'data' or 'error'.
        """
        headers = {
            "Content-Type": "application/sparql-query",
            "Accept": "application/sparql-results+json"
        }
        
        try:
            response = requests.post(self.endpoint, data=query, headers=headers, timeout=90)
            response.raise_for_status()
            
            return {
                "success": True,
                "data": response.json(),
                "error": None
            }
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "data": None,
                "error": "TIMEOUT"
            }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "data": None,
                "error": f"HTTP_ERROR: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "data": None,
                "error": f"UNEXPECTED: {str(e)}"
            }
        
    def normalize_results(self, json_result):
        bindings = json_result["results"]["bindings"]

        normalized = []
        for row in bindings:
            normalized.append(
                tuple(v["value"] for v in row.values())
            )

        return set(normalized)
    
    def jaccard_similarity(self, res1, res2):
        set1 = self.normalize_results(res1)
        set2 = self.normalize_results(res2)
        # print("Set 1:", set1)
        # print("Set 2:", set2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union else 1.
    def exact_match(self, res1, res2):
        set1 = self.normalize_results(res1)
        set2 = self.normalize_results(res2)
        return set1 == set2
    def precision_recall_f1(self, pred, gold):
        pred_set = self.normalize_results(pred)
        gold_set = self.normalize_results(gold)

        return self._precision_recall_f1(pred_set, gold_set)

    def answer_overlap_metrics(self, pred, gold):
        pred_vars = self._result_variables(pred)
        gold_vars = self._result_variables(gold)
        shared_vars = [var for var in pred_vars if var in gold_vars]

        if shared_vars:
            pred_set = self.normalize_results_for_variables(pred, shared_vars)
            gold_set = self.normalize_results_for_variables(gold, shared_vars)
            basis = "shared_variables:" + ",".join(shared_vars)
        else:
            pred_set = self.normalize_result_values(pred)
            gold_set = self.normalize_result_values(gold)
            basis = "all_values"

        intersection = len(pred_set & gold_set)
        union = len(pred_set | gold_set)
        jaccard = intersection / union if union else 1.0
        precision, recall, f1 = self._precision_recall_f1(pred_set, gold_set)

        return {
            "basis": basis,
            "jaccard": jaccard,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def normalize_results_for_variables(self, json_result, variables):
        bindings = json_result["results"]["bindings"]
        normalized = []
        for row in bindings:
            normalized.append(
                tuple(self._binding_value(row.get(var)) for var in variables)
            )
        return set(normalized)

    def normalize_result_values(self, json_result):
        bindings = json_result["results"]["bindings"]
        return {
            self._binding_value(value)
            for row in bindings
            for value in row.values()
        }

    @staticmethod
    def _result_variables(json_result):
        return json_result.get("head", {}).get("vars", [])

    @staticmethod
    def _binding_value(binding):
        if binding is None:
            return None
        return binding.get("value")

    @staticmethod
    def _precision_recall_f1(pred_set, gold_set):
        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = (2 * precision * recall / (precision + recall)
            if precision + recall else 0)

        return precision, recall, f1
    def extract_single_value(self, json_result):
        bindings = json_result["results"]["bindings"]

        if not bindings:
            return None

        # first row, first variable
        first_row = bindings[0]
        first_var = next(iter(first_row))
        return int(first_row[first_var]["value"])
    
    def exact_match_count(self, res1, res2):
        return self.extract_single_value(res1) == self.extract_single_value(res2)

    def measure_similarity(self, generated_query: str, actual_query: str) -> dict:
        """
        Always return a dict with consistent keys.
        Errors are captured in 'error' field, metrics set to None.
        """
        
        # --- Step 1: Execute both queries ---
        retrieved_docs_generated = self.post_query(generated_query)
        retrieved_docs_actual = self.post_query(actual_query)
        # print("Generated Query Result:", retrieved_docs_generated["data"])
        # print("Actual Query Result:", retrieved_docs_actual["data"])
        # --- Step 2: Check for execution errors ---
        if not retrieved_docs_actual["success"]:
            return self._error_result(
                error_type="actual_query_failed",
                error_msg=retrieved_docs_actual["error"]
            )
        
        if not retrieved_docs_generated["success"]:
            return self._error_result(
                error_type="generated_query_failed",
                error_msg=retrieved_docs_generated["error"]
            )
        
        # --- Step 3: Detect query types ---
        is_count_actual = "COUNT" in actual_query.upper()
        is_count_generated = "COUNT" in generated_query.upper()
        
        # --- Step 4: Handle query type mismatch ---
        if is_count_actual and not is_count_generated:
            return self._error_result(
                error_type="query_type_mismatch",
                error_msg="Expected COUNT query but generated standard query",
                query_type="mismatch"
            )
        
        # --- Step 5: Compute metrics with try-except ---
        try:
            if is_count_actual:
                count_result = self.exact_match_count(
                    retrieved_docs_generated["data"],
                    retrieved_docs_actual["data"]
                )
                llm_judgement = self.llm_judge.judge(
                    question=self.question,
                    gold_results=retrieved_docs_actual["data"],
                    generated_results=retrieved_docs_generated["data"],
                ) if self.llm_judge else {
                    "decision": "No",
                    "reason": "LLM judge unavailable.",
                }
                return {
                    "query_type": "count",
                    "error": None,
                    "exact_match_count": float(count_result) if count_result is not None else None,
                    "jaccard_similarity": None,
                    "is_exact_results": None,
                    "precision": None,
                    "recall": None,
                    "f1_score": None,
                    "answer_overlap_basis": None,
                    "answer_overlap_jaccard": None,
                    "answer_overlap_precision": None,
                    "answer_overlap_recall": None,
                    "answer_overlap_f1": None,
                    "llm_equivalence_decision": llm_judgement["decision"],
                    "llm_equivalence_reason": llm_judgement["reason"],
                }
            else:
                jaccard = self.jaccard_similarity(
                    retrieved_docs_generated["data"],
                    retrieved_docs_actual["data"]
                )
                exact = self.exact_match(
                    retrieved_docs_generated["data"],
                    retrieved_docs_actual["data"]
                )
                p, r, f1 = self.precision_recall_f1(
                    retrieved_docs_generated["data"],
                    retrieved_docs_actual["data"]
                )
                answer_overlap = self.answer_overlap_metrics(
                    retrieved_docs_generated["data"],
                    retrieved_docs_actual["data"]
                )
                llm_judgement = self.llm_judge.judge(
                    question=self.question,
                    gold_results=retrieved_docs_actual["data"],
                    generated_results=retrieved_docs_generated["data"],
                ) if self.llm_judge else {
                    "decision": "No",
                    "reason": "LLM judge unavailable.",
                }
                
                return {
                    "query_type": "standard",
                    "error": None,
                    "exact_match_count": None,
                    "jaccard_similarity": float(jaccard) if jaccard is not None else None,
                    "is_exact_results": float(exact) if exact is not None else None,
                    "precision": float(p) if p is not None else None,
                    "recall": float(r) if r is not None else None,
                    "f1_score": float(f1) if f1 is not None else None,
                    "answer_overlap_basis": answer_overlap["basis"],
                    "answer_overlap_jaccard": float(answer_overlap["jaccard"]),
                    "answer_overlap_precision": float(answer_overlap["precision"]),
                    "answer_overlap_recall": float(answer_overlap["recall"]),
                    "answer_overlap_f1": float(answer_overlap["f1"]),
                    "llm_equivalence_decision": llm_judgement["decision"],
                    "llm_equivalence_reason": llm_judgement["reason"],
                }
        
        except Exception as e:
            return self._error_result(
                error_type="metric_calculation_failed",
                error_msg=str(e)
            )

    def _error_result(self, error_type: str, error_msg: str, query_type: str = "error") -> dict:
        """Helper to return consistent error structure."""
        return {
            "query_type": query_type,
            "error": f"{error_type}: {error_msg}",
            "exact_match_count": None,
            "jaccard_similarity": None,
            "is_exact_results": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
            "answer_overlap_basis": None,
            "answer_overlap_jaccard": None,
            "answer_overlap_precision": None,
            "answer_overlap_recall": None,
            "answer_overlap_f1": None,
        }
    
class QueryValidator:
    def __init__(
        self,
        endpoint: str="https://sparql.knowledgehub.nfdi4earth.de",
        llm=None,
        llm_judge_prompt: str | None = None,
    ):
        self.endpoint = endpoint
        # store invalid queries: {question_type: [ (question, query, error) ]}
        self.invalid_queries = defaultdict(list)
        # count total valid/invalid queries
        self.stats = defaultdict(int)
        self.query_similarity = QuerySimilarity()
        self.syntax_validator = SyntaxValidator()
        self.llm_judge = LLMResultEquivalenceJudge(llm=llm, prompt_template=llm_judge_prompt)
        self.result_similarity = ResultSimilarity(endpoint=endpoint, llm_judge=self.llm_judge)


    def process_queries(self,  generated_query:str , actual_query: str, question: str="hi", question_type: str = "default"):
        """Validate a query and track errors."""
        syntax_validation_result = self.syntax_validator.validate(generated_query)
        if not isinstance(actual_query, str) or not actual_query.strip():
            return (
                {
                    "structure_similarity": {},
                    "language_similarity": {},
                },
                syntax_validation_result,
                {
                    "query_type": "skipped",
                    "error": "Gold query is empty - evaluation metrics not computed",
                    "exact_match_count": None,
                    "jaccard_similarity": None,
                    "is_exact_results": None,
                    "precision": None,
                    "recall": None,
                    "f1_score": None,
                    "answer_overlap_basis": None,
                    "answer_overlap_jaccard": None,
                    "answer_overlap_precision": None,
                    "answer_overlap_recall": None,
                    "answer_overlap_f1": None,
                },
                syntax_validation_result.grammar_ok,
            )

        sparql_similarities = self.query_similarity.measure_similarities(
            generated_query, actual_query
        )
        #  FIX: Initialize as proper error dict, not tuple
        results_similarities = {
            "query_type": "skipped",
            "error": "Syntax validation failed - metrics not computed",
            "exact_match_count": None,
            "jaccard_similarity": None,
            "is_exact_results": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
            "answer_overlap_basis": None,
            "answer_overlap_jaccard": None,
            "answer_overlap_precision": None,
            "answer_overlap_recall": None,
            "answer_overlap_f1": None,
        }
        if(syntax_validation_result.grammar_ok): #and syntax_validation_result.preparation_ok commented for now
            self.result_similarity.question = question
            results_similarities = self.result_similarity.measure_similarity(generated_query=generated_query, actual_query=actual_query)
        is_valid = syntax_validation_result.grammar_ok #and syntax_validation_result.preparation_ok
        return sparql_similarities, syntax_validation_result , results_similarities, is_valid
        # # # Prepare metadata for this run
        # metadata = {
        #     "model_name": "model_name",
        #     "schema_mode": "schema_mode.value",
        #     "schema_label": "schema_label",
        #     "question_id": "index",
        #     "run_timestamp": pd.Timestamp.now().isoformat(),
        # }

        # # Serialize to flat row
        # result_row = ResultSerializer.serialize_row(
        #     question=question,
        #     generated_sparql=generated_query,
        #     sparql_similarity=sparql_similarities,
        #     validation_result=syntax_validation_result,
        #     execution_time=0,
        #     is_valid_final=is_valid,
        #     result_similarity=results_similarities,
        #     metadata=metadata
        # )
        # print(result_row)

    def analyse_generated_queries(self, file_path:str, model_files=[ "LLAMA", "CommandRPlus", "Qwen_instrcut"]):
        """
        file_path is a csv file where there is at least 
        1. question 
        2. true query 
        3. generated query 
        4. question type
        """
        # All model output files

        ##open file location 
        ##Extract all these information 
        ##Run process queries for each model
        df = pd.read_csv("260202_extracted_question.csv", sep='\t')
        print(df.head())

        #self.process_queries()
        pass


    def generate_report(self, file_path: str = None):
        """Generate a report of invalid queries by question type."""
        report = {
            "total_valid": self.stats.get('valid', 0),
            "total_invalid": self.stats.get('invalid', 0),
            "invalid_queries_by_type": dict(self.invalid_queries)
        }

        # Optionally write to a JSON file
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=4)
        
        return report





# utils/result_serializer.py
import json
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional

class ResultSerializer:
    """Convert complex evaluation results into flat, CSV-friendly format."""
    
    @staticmethod
    def flatten_similarity(sim_dict: Dict[str, Any], prefix: str = "sim") -> Dict[str, float]:
        """Flatten nested similarity metrics into dot-notation columns."""
        flat = {}
        
        # Top-level metrics
        if 'content_similarity' in sim_dict:
            flat[f"{prefix}_content"] = float(sim_dict['content_similarity'])
        
        # Structure similarity sub-metrics
        if 'structure_similarity' in sim_dict:
            struct = sim_dict['structure_similarity']
            for key in ['string_similarity', 'graph_similarity', 'semantic_similarity', 'combined_similarity']:
                if key in struct:
                    val = struct[key]
                    # Handle numpy types
                    if isinstance(val, (np.floating, np.integer)):
                        val = val.item()
                    flat[f"{prefix}_struct_{key}"] = float(val)
        
        # Language similarity (keep vectors as JSON if needed)
        if 'language_similarity' in sim_dict:
            lang = sim_dict['language_similarity']
            for key in ['frequency_similarity', 'boolean_similarity']:
                if key in lang:
                    flat[f"{prefix}_lang_{key}"] = float(lang[key])
            # Optional: store vectors as JSON strings (useful for debugging, heavy for plotting)
            # for key in ['frequency_vector_q1', 'frequency_vector_q2', 'boolean_vector_q1', 'boolean_vector_q2']:
            #     if key in lang:
            #         flat[f"{prefix}_lang_{key}"] = json.dumps(lang[key].tolist() if isinstance(lang[key], np.ndarray) else lang[key])
        
        return flat

    @staticmethod
    def flatten_validation_result(val_result: Any, prefix: str = "val") -> Dict[str, Any]:
        """Extract key fields from ValidationResult object."""
        return {
            f"{prefix}_grammar_ok": bool(val_result.grammar_ok),
            f"{prefix}_grammar_error": str(val_result.grammar_error) if val_result.grammar_error else None,
            f"{prefix}_preparation_ok": bool(val_result.preparation_ok),
            f"{prefix}_preparation_error": str(val_result.preparation_error) if val_result.preparation_error else None,
        }


    @staticmethod
    def flatten_query_metrics(sim_metrics: dict, prefix: str = "metric") -> dict:
        """Flatten similarity metrics with safe None handling + error capture."""
        flat = {}
        
        # Query type and error info
        flat[f"{prefix}_query_type"] = sim_metrics.get("query_type", "unknown")
        flat[f"{prefix}_error"] = sim_metrics.get("error")  # Will be None if no error
        
        # Map metric names (handle None -> np.nan)
        metric_mapping = {
            "exact_match_count": f"{prefix}_exact_count",
            "jaccard_similarity": f"{prefix}_jaccard",
            "is_exact_results": f"{prefix}_exact_match",
            "precision": f"{prefix}_precision",
            "recall": f"{prefix}_recall",
            "f1_score": f"{prefix}_f1",
            "answer_overlap_jaccard": f"{prefix}_answer_overlap_jaccard",
            "answer_overlap_precision": f"{prefix}_answer_overlap_precision",
            "answer_overlap_recall": f"{prefix}_answer_overlap_recall",
            "answer_overlap_f1": f"{prefix}_answer_overlap_f1",
        }
        
        for key, col_name in metric_mapping.items():
            val = sim_metrics.get(key)
            flat[col_name] = float(val) if val is not None else np.nan

        flat[f"{prefix}_answer_overlap_basis"] = sim_metrics.get("answer_overlap_basis")
        flat[f"{prefix}_llm_equivalence_decision"] = sim_metrics.get("llm_equivalence_decision")
        flat[f"{prefix}_llm_equivalence_reason"] = sim_metrics.get("llm_equivalence_reason")
        
        return flat
    
    @staticmethod
    def flatten_metrics(        
        sparql_similarity: Dict,      # structure/content/language similarity
        validation_result: Any,        # ValidationResult object
        result_similarity: Dict,           # jaccard, precision, recall, f1, etc.

    ) -> Dict[str, Any]:
        """Combine all evaluation outputs into a single flat dictionary for one row."""
        
        row = {
            # Core identifiers
         
            **ResultSerializer.flatten_similarity(sparql_similarity),
            
            # Validation results
            **ResultSerializer.flatten_validation_result(validation_result),
            **ResultSerializer.flatten_query_metrics(result_similarity),
         
        }
        
        return row

    @staticmethod
    def serialize_row(
        question: str,
        generated_sparql: str,
        actual_sparql: str,
        sparql_similarity: Dict,      # structure/content/language similarity
        validation_result: Any,        # ValidationResult object
        result_similarity: Dict,           # jaccard, precision, recall, f1, etc.
        metadata: Dict[str, str],
        execution_time: Optional[float] = None,  # Optional timing
        is_valid_final: Optional[bool] = None    # Optional validity flag
    ) -> Dict[str, Any]:
        """Combine all evaluation outputs into a single flat dictionary for one row."""
        
        row = {
            # Core identifiers
            "question": question,
            "generated_sparql": generated_sparql,
            "actual_sparql": actual_sparql,
            
            # Metadata for filtering/grouping
            **metadata,  # e.g., {'model_name': 'llama3', 'schema_mode': 'rag', 'question_id': 42}
            

                # Optional top-level fields
            "execution_time_sec": float(execution_time) if execution_time is not None else np.nan,
            "is_valid_final": bool(is_valid_final) if is_valid_final is not None else np.nan,
            # Flattened similarity metrics
            **ResultSerializer.flatten_similarity(sparql_similarity),
            
            # Validation results
            **ResultSerializer.flatten_validation_result(validation_result),
            **ResultSerializer.flatten_query_metrics(result_similarity),
         
        }
        
        return row



# class kkResultSerializer:
#     #Case 1 --> generate query is already updated 
#     # receive conditions, model_name and schema_mode
#     #1. read both actual and generated query 
#     # 2. run query validator for each 
#     # 3. serialize the results 
#     # 4. update its row metric 
#     pass 

# q1 = """
# PREFIX n4e: <http://nfdi4earth.de/ontology/>

# SELECT *
# WHERE {
#   ?subject a n4e:EntityType.
# }
# """

# q2 = """
# PREFIX n4e: <http://nfdi4earth.de/ontology/>

# SELECT *
# WHERE {
#   ?subject a n4e:EntityType.
# }
# """




# # q1  ="""PREFIX n4e: <http://nfdi4earth.de/ontology/>
# # PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

# # SELECT (COUNT(DISTINCT ?predicate) AS ?count)
# # WHERE {
# #   ?entity rdf:type n4e:LHBArticle ;
# #           ?predicate ?value .
# #   FILTER(?predicate != rdf:type)
# # } 
# # """


# # q2  ="""PREFIX n4e: <http://nfdi4earth.de/ontology/>
# # PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

# # SELECT (COUNT(DISTINCT ?predicate) AS ?count)
# # WHERE {
# #   ?entity rdf:type n4e:LHBArticle ;
# #           ?predicate ?value .
# #   FILTER(?predicate != rdf:type)
# # }"""
# query_validator  = QueryValidator()
# query_validator.process_queries(q1, q2)
