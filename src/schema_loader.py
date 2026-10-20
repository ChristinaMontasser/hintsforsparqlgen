import os
import json
import pickle
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional
from collections import defaultdict

# --- Configuration Enum ---
class SchemaMode(Enum):
    FULL_SCHEMA = "full"          # Load whole TTL/OWL (Risk: Token Limit)
    SMALL_SCHEMA = "small"        # Load pre-reduced TTL/OWL
    RAG_TRIPLES = "rag"           # Load relevant triples via Vector Search
    NER = "ner"                   # Build schema hints via LLM-based extraction + lexicon mapping
    SCHEMA_PATH = "schema_path"   # Build a domain/range-aware graph plan via LLM + lexicon refinement
    REASONING_PATH = "reasoning_path"       # Build a reasoning path from serialized schema paths
    LEXICON = "lexicon"           # Load reduced lexicon as an allowed URI list
    ZERO_SHOT = "none"            # Load nothing


PREFIX_URIS = {
    "dcat": "http://www.w3.org/ns/dcat#",
    "dcatde": "http://dcat-ap.de/def/dcatde/",
    "dct": "http://purl.org/dc/terms/",
    "envthes": "http://vocabs.lter-europe.net/EnvThes/",
    "foaf": "http://xmlns.com/foaf/0.1/",
    "geo": "http://www.opengis.net/ont/geosparql#",
    "locn": "http://www.w3.org/ns/locn#",
    "m4i": "http://w3id.org/nfdi4ing/metadata4ing#",
    "n4e": "http://nfdi4earth.de/ontology/",
    "nfdi": "https://nfdi.fiz-karlsruhe.de/ontology/",
    "org": "http://www.w3.org/ns/org#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "schema": "http://schema.org/",
    "sf": "http://www.opengis.net/ont/sf#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "sosa": "http://www.w3.org/ns/sosa/",
    "hyobs": "https://w3id.org/hmontology/",
    "vcard": "http://www.w3.org/2006/vcard/ns#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
}

# --- RAG Engine (Handles Embedding & Index Persistence) ---
class OntologyVectorStore:
    def __init__(self, index_path: str, chunks_path: str, k = 5, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.index_path = index_path
        self.k = k
        self.chunks_path = chunks_path
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []

    def build_and_save(self, ontology_path: str):
        """Run this ONCE to create the index and chunks files."""
        import faiss
        from rdflib import Graph

        print(f"Building Vector Store from {ontology_path}...")
        
        # 1. Parse & Chunk
        graph = Graph()
        graph.parse(ontology_path, format="turtle") # or xml/owl
        self.chunks = self._group_by_subject(graph)
        
        # 2. Embed
        embeddings = self.model.encode(self.chunks, convert_to_numpy=True, show_progress_bar=True)
        embeddings = embeddings.astype("float32")
        
        # 3. Build FAISS
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        # 4. Persist to Disk
        faiss.write_index(self.index, self.index_path)
        with open(self.chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
            
        print(f"Vector Store saved to {self.index_path} and {self.chunks_path}")

    def load(self):
        """Load existing index and chunks from disk."""
        import faiss

        if not os.path.exists(self.index_path) or not os.path.exists(self.chunks_path):
            raise FileNotFoundError("Vector store not found. Run build_and_save() first.")
        
        self.index = faiss.read_index(self.index_path)
        with open(self.chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)
        print("Vector Store loaded from disk.")

    def retrieve(self, query: str) -> List[str]:
        
        if self.index is None or not self.chunks:
            raise RuntimeError("Vector store not loaded.")
            
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype("float32")
        distances, indices = self.index.search(query_embedding,  self.k )
        
        # Return text chunks corresponding to indices
        return [self.chunks[i] for i in indices[0] if i < len(self.chunks)]

    def _group_by_subject(self, graph: Any) -> List[str]:
        """Helper to convert RDF to text chunks."""
        subject_map = defaultdict(list)
        for subj, pred, obj in graph:
            subject_map[str(subj)].append(f"{pred} {obj}")
        
        chunks = []
        for subj, statements in subject_map.items():
            combined = f"Subject: {subj}\n" + "\n".join(statements)
            chunks.append(combined)
        return chunks

# --- Main Loader Interface ---
class SchemaLoader:
    def __init__(self, mode: SchemaMode, config: dict):
        """
        config expects keys: 
          - 'full_schema_path'
          - 'small_schema_path'
          - 'lexicon_schema_path'
          - 'vector_store' (Instance of OntologyVectorStore, required if mode is RAG)
        """
        self.mode = mode
        self.config = config
        self._schema_cache = {} # Cache text schemas to avoid re-reading files

    def get_context(self, query: Optional[str] = None, question_id: Optional[str] = None) -> str:
        """
        Returns the schema context string based on the configured mode.
        If RAG mode, 'query' is required to retrieve relevant triples.
        """
        if self.mode == SchemaMode.ZERO_SHOT:
            return ""
        
        if self.mode == SchemaMode.FULL_SCHEMA:
            print(self.config['full_schema_path'])
            return self._load_file(self.config['full_schema_path'])
        
        if self.mode == SchemaMode.SMALL_SCHEMA:
            return self._load_file(self.config['small_schema_path'])

        if self.mode == SchemaMode.LEXICON:
            return self._load_lexicon_context(self.config['lexicon_schema_path'])
        
        if self.mode == SchemaMode.RAG_TRIPLES:
            if not query:
                raise ValueError("Query is required for RAG mode to retrieve context.")
            vs = self.config.get('vector_store')
            if not vs:
                raise ValueError("VectorStore instance required in config for RAG mode.")
            
            chunks = vs.retrieve(query)
            print(chunks)
            return "\n\n".join(chunks)

        if self.mode == SchemaMode.NER:
            if not query:
                raise ValueError("Query is required for NER mode to retrieve context.")
            ner_builder = self.config.get('ner_builder')
            if not ner_builder:
                raise ValueError("NER context builder instance required in config for NER mode.")
            return ner_builder.get_context(query, question_id=question_id)

        if self.mode == SchemaMode.SCHEMA_PATH:
            if not query:
                raise ValueError("Query is required for schema_path mode to retrieve context.")
            schema_path_builder = self.config.get('schema_path_builder')
            if not schema_path_builder:
                raise ValueError("Schema path builder instance required in config for schema_path mode.")
            return schema_path_builder.get_context(query, question_id=question_id)

        if self.mode == SchemaMode.REASONING_PATH:
            if not query:
                raise ValueError("Query is required for reasoning_path mode to retrieve context.")
            reasoning_path_builder = self.config.get('reasoning_path_builder')
            if not reasoning_path_builder:
                raise ValueError("Reasoning path builder instance required in config for reasoning_path mode.")
            return reasoning_path_builder.get_context(query, question_id=question_id)
        
        return ""

    def _load_file(self, path: str) -> str:
        if self._schema_cache and self.mode != SchemaMode.RAG_TRIPLES:
            # Simple cache optimization for static files
            if self.mode == SchemaMode.FULL_SCHEMA and 'full' in self._schema_cache:
                return self._schema_cache['full']
            if self.mode == SchemaMode.SMALL_SCHEMA and 'small' in self._schema_cache:
                return self._schema_cache['small']

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Update cache
        if self.mode == SchemaMode.FULL_SCHEMA:
            self._schema_cache['full'] = content
        elif self.mode == SchemaMode.SMALL_SCHEMA:
            self._schema_cache['small'] = content
            
        return content

    def _load_lexicon_context(self, path: str) -> str:
        if 'lexicon' in self._schema_cache:
            return self._schema_cache['lexicon']

        lexicon = json.loads(Path(path).read_text(encoding="utf-8"))
        context = self._format_lexicon_context(lexicon)
        self._schema_cache['lexicon'] = context
        return context

    def _format_lexicon_context(self, lexicon: dict[str, Any]) -> str:
        classes = self._valid_lexicon_entries(lexicon.get("classes", []), "class_uri")
        relations = self._valid_lexicon_entries(lexicon.get("relations", []), "slot_uri")
        entities = self._valid_lexicon_entries(lexicon.get("entities", []), "uri")

        lines = [
            "Reduced lexicon schema context.",
            "Generate a SPARQL query for the input question using only this schema context.",
            "Use only class_uri values as rdf:type objects.",
            "Use only slot_uri values as predicates.",
            "Do not use ids or labels as SPARQL IRIs unless they are also listed as URI values.",
            "Declare every prefix used in the generated SPARQL query.",
            "",
            "Prefix declarations:",
        ]
        lines.extend(self._format_prefix_declarations(classes + relations + entities))

        lines.extend([
            "",
            "Allowed class_uri values:",
        ])
        lines.extend(self._format_lexicon_section(classes, "class_uri"))

        lines.extend([
            "",
            "Allowed slot_uri values:",
        ])
        lines.extend(self._format_lexicon_section(relations, "slot_uri"))

        if entities:
            lines.extend([
                "",
                "Allowed named entity uri values:",
            ])
            lines.extend(self._format_lexicon_section(entities, "uri"))

        return "\n".join(lines)

    @staticmethod
    def _valid_lexicon_entries(entries: Any, uri_key: str) -> list[dict[str, Any]]:
        if not isinstance(entries, list):
            return []

        return [
            entry for entry in entries
            if isinstance(entry, dict)
            and isinstance(entry.get(uri_key), str)
            and entry[uri_key].strip()
        ]

    def _format_prefix_declarations(self, entries: list[dict[str, Any]]) -> list[str]:
        prefixes = {"rdf"}
        for entry in entries:
            for key in ("class_uri", "slot_uri", "uri"):
                value = entry.get(key)
                if isinstance(value, str) and ":" in value:
                    prefixes.add(value.split(":", 1)[0])

        lines = []
        for prefix in sorted(prefixes):
            uri = PREFIX_URIS.get(prefix)
            if uri:
                lines.append(f"PREFIX {prefix}: <{uri}>")
            else:
                lines.append(f"# Prefix {prefix}: URI not configured")
        return lines

    @staticmethod
    def _format_lexicon_section(entries: list[dict[str, Any]], uri_key: str) -> list[str]:
        if not entries:
            return ["- None"]

        lines = []
        for entry in entries:
            label = entry.get("label") or entry.get("id") or "unlabeled"
            entry_id = entry.get("id")
            uri = entry[uri_key]
            if entry_id and entry_id != label:
                lines.append(f"- {label} ({entry_id}): {uri}")
            else:
                lines.append(f"- {label}: {uri}")
        return lines
