import re
import math
from collections import defaultdict
from itertools import combinations
import numpy as np
from scipy.optimize import linear_sum_assignment

#########################
## Implementation of Similarity measure from the following Paper
#Wang, M., Chen, K., Xiao, G. et al. Explaining similarity for SPARQL queries. 
# World Wide Web 24, 1813–1835 (2021). https://doi.org/10.1007/s11280-021-00886-3
#########################





# --------------------------------------------------
# Utility: Levenshtein Distance
# --------------------------------------------------
def levenshtein_distance( s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def cosine_similarity(v1, v2):
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)


class Semantic_Similarity:
    def __init__(self, ):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    def compute_embeddings(self,  Q1, Q2):
        v1, v2 = self.embedding_model.encode( Q1), self.embedding_model.encode(Q2)
        return v1, v2
    
    def measure_similarity(self, Q1, Q2):
        v1, v2= self.compute_embeddings(Q1, Q2)
        return cosine_similarity(v1, v2)
         


class Structure_Similarity:

    def __init__(self):
        self.semantic_similarity = Semantic_Similarity()

    # --------------------------------------------------
    # Feature 1: String-based Structural Feature
    # --------------------------------------------------
    def _normalize_template(self, query: str) -> str:
        # Remove PREFIX lines
        query = re.sub(r'PREFIX\s+[^\n]+\n', '', query, flags=re.IGNORECASE)

        # Replace variables, IRIs, literals
        query = re.sub(r'\?[a-zA-Z_][\w]*', '_VAR_', query)
        query = re.sub(r'<[^>]*>', '_IRI_', query)
        query = re.sub(r'"[^"]*"', '_LIT_', query)

        # Normalize whitespace
        query = re.sub(r'\s+', ' ', query).strip()

        return query


    def _string_structure_similarity(self, q1: str, q2: str) -> float:
        t1 = self._normalize_template(q1)
        t2 = self._normalize_template(q2)

        sed = levenshtein_distance(t1, t2)
        max_len = max(len(t1), len(t2))

        if max_len == 0:
            return 1.0

        # Similarity = 1 - normalized edit distance
        return 1 - (sed / max_len)


    # --------------------------------------------------
    # Feature 1: String-based Structural Feature
    # --------------------------------------------------
    def _semantic_structure_similarity(self, q1: str, q2: str) -> float:
        return self.semantic_similarity.measure_similarity(q1, q2)
    
    # --------------------------------------------------
    # Extract Basic Graph Patterns (BGPs)
    # --------------------------------------------------
    def _extract_bgp_blocks(self, query: str):
        where_match = re.search(r'WHERE\s*{(.*)}', query, re.IGNORECASE | re.DOTALL)
        if not where_match:
            return []

        where_body = where_match.group(1)

        # Split on OPTIONAL/UNION etc (simple heuristic)
        bgps = re.split(r'OPTIONAL|UNION', where_body, flags=re.IGNORECASE)

        return bgps


    def _extract_triples(self, bgp_block: str):
        triples = []
        lines = bgp_block.split('.')
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 3:
                triples.append(tuple(parts[:3]))
        return triples


    # --------------------------------------------------
    # Feature 2: Graph-based Structural Feature
    # --------------------------------------------------
    def _graph_feature_vector(self, query: str):
        bgps = self._extract_bgp_blocks(query)

        triple_count = 0
        projection_count = 0
        vertex_degree = defaultdict(int)

        for bgp in bgps:
            triples = self._extract_triples(bgp)
            triple_count += len(triples)

            for s, p, o in triples:
                vertex_degree[s] += 1
                vertex_degree[o] += 1

        # Projection variables
        select_match = re.search(r'SELECT\s+(.*?)\s+WHERE', query, re.IGNORECASE | re.DOTALL)
        if select_match:
            projection_count = len(re.findall(r'\?[a-zA-Z_]\w*', select_match.group(1)))

        # Join statistics
        star_joins = 0
        path_joins = 0
        hybrid_joins = 0
        high_degree_nodes = 0

        for node, deg in vertex_degree.items():
            if deg >= 2:
                star_joins += 1
            if deg == 2:
                path_joins += 1
            if deg > 2:
                hybrid_joins += 1
            if deg >= 3:
                high_degree_nodes += 1

        feature_vector = [
            triple_count,            # 1. #triple patterns
            len(bgps),               # 2. #BGPs
            projection_count,        # 3. #projections
            star_joins,              # 4
            path_joins,              # 5
            hybrid_joins,            # 6
            high_degree_nodes,       # 7
            len(vertex_degree),      # 8 #vertices
            sum(vertex_degree.values()), # 9 total degree
            1                        # 10 avoid zero vector
        ]

        return feature_vector


    def _graph_structure_similarity(self, q1: str, q2: str) -> float:
        v1 = self._graph_feature_vector(q1)
        v2 = self._graph_feature_vector(q2)
        return cosine_similarity(v1, v2)


    # --------------------------------------------------
    # Final Combined Function
    # --------------------------------------------------
    def measure_similarity(self,q1: str, q2: str):
        string_sim = self._string_structure_similarity(q1, q2)
        semantic_similarity = self._semantic_structure_similarity(q1, q2)
        graph_sim = self._graph_structure_similarity(q1, q2)

        return {
            "string_similarity": string_sim,
            "graph_similarity": graph_sim,
            "semantic_similarity": semantic_similarity,
            "combined_similarity": ( semantic_similarity+ graph_sim) / 2
        }


class Language_Similarity:
      # ----------------------------------------------
    # SPARQL Language Keywords (editable)
    # ----------------------------------------------
    def __init__(self):
        
        self.LANGUAGE_KEYWORDS = [
            "SELECT",
            "ASK",
            "DISTINCT",
            "LIMIT",
            "OFFSET",
            "FILTER",
            "UNION",
            "OPTIONAL",
            "ORDER BY",
            "GROUP BY",
            "HAVING",
            "VALUES",
            "BIND"
        ]



    # ----------------------------------------------
    # Preprocess Query (case normalization)
    # ----------------------------------------------
    def preprocess_query(self, query: str) -> str:
        return re.sub(r'\s+', ' ', query.upper())


    # ----------------------------------------------
    # Feature 6: Frequency-Based Language Vector
    # ----------------------------------------------
    def language_vector_frequency(self, query: str):
        query = self.preprocess_query(query)
        vector = []

        for keyword in self.LANGUAGE_KEYWORDS:
            if " " in keyword:  # multi-word operator (ORDER BY, GROUP BY)
                pattern = keyword
            else:
                pattern = r'\b' + keyword + r'\b'

            count = len(re.findall(pattern, query))
            vector.append(count)

        vector.append(1)  # Avoid zero vector
        return vector


    # ----------------------------------------------
    # Feature 7: Boolean Language Vector
    # ----------------------------------------------
    def language_vector_boolean(self, query: str):
        query = self.preprocess_query(query)
        vector = []

        for keyword in self.LANGUAGE_KEYWORDS:
            if " " in keyword:
                exists = 1 if keyword in query else 0
            else:
                exists = 1 if re.search(r'\b' + keyword + r'\b', query) else 0

            vector.append(exists)

        vector.append(1)  # Avoid zero vector
        return vector


    # ----------------------------------------------
    # Language Similarity Between Two Queries
    # ----------------------------------------------
    def language_similarity(self, q1: str, q2: str, mode="frequency"):
        if mode == "frequency":
            v1 = self.language_vector_frequency(q1)
            v2 = self.language_vector_frequency(q2)
        elif mode == "boolean":
            v1 = self.language_vector_boolean(q1)
            v2 = self.language_vector_boolean(q2)
        else:
            raise ValueError("mode must be 'frequency' or 'boolean'")

        return cosine_similarity(v1, v2)


    # ----------------------------------------------
    # Full Feature Output
    # ----------------------------------------------
    def measure_similarity(self, q1: str, q2: str):
        return {
            "frequency_similarity": self.language_similarity(q1, q2, mode="frequency"),
            "boolean_similarity":  self.language_similarity(q1, q2, mode="boolean"),
            "frequency_vector_q1":  self.language_vector_frequency(q1),
            "frequency_vector_q2":  self.language_vector_frequency(q2),
            "boolean_vector_q1":  self.language_vector_boolean(q1),
            "boolean_vector_q2":  self.language_vector_boolean(q2),
        }



class Content_Similarity:
    def __init__(self):
        pass

    def normalized_sed(self, x1, x2):
        return levenshtein_distance(x1, x2) / (max(len(x1), len(x2)) + 1)


    # ============================================================
    # Item Type Classification
    # ============================================================

    def classify(self, x):
        if x.startswith("?"):
            return "V"
        if x.startswith('"'):
            return "L"
        return "I"


    # ============================================================
    # similarityFuncForItem (Eq 2,3,4,5)
    # ============================================================

    def similarity_item(self, x1, x2, mode="naive", k=0.5):
        t1, t2 = self.classify(x1), self.classify(x2)

        # Different type → 0
        if t1 != t2:
            return 0.0

        if mode == "naive":  # Feature 3 (Eq 2)
            return 1.0 if x1 == x2 else 0.0

        sed_value = self.normalized_sed(x1, x2)

        if mode == "sed":  # Feature 4 (Eq 3)
            return 1 - sed_value

        if mode == "weighted_sed":  # Feature 5 (Eq 4–5)
            if t1 == "V":
                return (1 - sed_value) * k
            else:
                return 1 - sed_value

        raise ValueError("mode must be naive | sed | weighted_sed")


    # ============================================================
    # Algorithm 3 — similarityFuncForTriple
    # ============================================================

    def similarity_triple(self, t1, t2, mode="naive", k=0.5):
        scores = []
        for i in range(3):  # s, p, o
            scores.append(self.similarity_item(t1[i], t2[i], mode, k))
        return sum(scores) / 3.0


    # ============================================================
    # Extract Operator Blocks (Algorithm 1 line 1–3)
    # ============================================================

    def get_operator_blocks(self, query):
        blocks = {}
        query = re.sub(r'\s+', ' ', query)

        where_match = re.search(r'WHERE\s*{(.*)}', query, re.IGNORECASE)
        if not where_match:
            return blocks

        where_body = where_match.group(1)

        # OPTIONAL
        blocks["OPTIONAL"] = re.findall(r'OPTIONAL\s*{([^}]*)}', where_body, re.IGNORECASE)

        # UNION
        unions = re.findall(r'{([^}]*)}\s*UNION\s*{([^}]*)}', where_body, re.IGNORECASE)
        blocks["UNION"] = []
        for u1, u2 in unions:
            blocks["UNION"].append(u1)
            blocks["UNION"].append(u2)

        # FILTER
        blocks["FILTER"] = re.findall(r'FILTER\s*\(([^)]*)\)', where_body, re.IGNORECASE)

        # MAIN block (entire WHERE content)
        blocks["MAIN"] = [where_body]
        #return 
        return blocks


    def get_triples_from_block(self, block):
        triples = []
        for stmt in block.split('.'):
            tokens = stmt.strip().split()
            if len(tokens) >= 3:
                triples.append(tuple(tokens[:3]))
        return triples


    # ============================================================
    # Algorithm 2 — Mapping Generation (Hungarian)
    # ============================================================

    def mapping_generation(self, T1, T2, mode="naive", k=0.5):

        if not T1 or not T2:
            return []

        cost_matrix = np.zeros((len(T1), len(T2)))

        for i in range(len(T1)):
            for j in range(len(T2)):
                sim = self.similarity_triple(T1[i], T2[j], mode, k)
                cost_matrix[i][j] = 1 - sim

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_pairs = []
        for r, c in zip(row_ind, col_ind):
            matched_pairs.append((T1[r], T2[c]))

        return matched_pairs


    # ============================================================
    # Algorithm 1 — Content Similarity
    # ============================================================

    def measure_similarity(self, Q1, Q2, mode="naive", k=0.5):

        Op1 = self.get_operator_blocks(Q1)
        Op2 = self.get_operator_blocks(Q2)
        #print(Op1)
        Op = set()

        for op in Op1:
            if Op1[op]:
                Op.add(op)

        for op in Op2:
            if Op2[op]:
                Op.add(op)

        scores_per_operator = []
        for op in Op:

            if op in Op1 and op in Op2 and Op1[op] and Op2[op]:

                T1 = []
                for b in Op1[op]:
                    T1.extend(self.get_triples_from_block(b))

                T2 = []
                for b in Op2[op]:
                    T2.extend(self.get_triples_from_block(b))

                matched = self.mapping_generation(T1, T2, mode, k)

                scores_per_triplepair = []
                for (t1, t2) in matched:
                    score = self.similarity_triple(t1, t2, mode, k)
                    scores_per_triplepair.append(score)

                maxLen = max(len(T1), len(T2))

                if maxLen > 0:
                    scores_per_operator.append(sum(scores_per_triplepair) / maxLen)
                else:
                    scores_per_operator.append(0)

            else:
                scores_per_operator.append(0)

        if not scores_per_operator:
            return 0.0
        #print(len(scores_per_operator))
        return sum(scores_per_operator) / len(scores_per_operator)

from sentence_transformers import SentenceTransformer




q1 = """
PREFIX n4e: <http://nfdi4earth.de/ontology/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX schema: <http://schema.org/>

SELECT *
WHERE {
  ?subject rdf:type n4e:LHBArticle;
   			schema:name ?name; 
      		schema:audience ?audience.
}
"""

q2 = """
PREFIX n4e: <http://nfdi4earth.de/ontology/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX schema: <http://schema.org/>

SELECT *
WHERE {
  ?subject rdf:type n4e:Article;
   			schema:name ?name; 
      		schema:audience ?audience.
}

"""


# structure_similarity = Structure_Similarity()
# language_similarity = Language_Similarity()
# content_similarity = Content_Similarity()
# print(f"Structure Similarity: {structure_similarity.measure_similarity(q1, q2)}")
# print(f"Language Similarity: {language_similarity.measure_similarity(q1, q2)}")
# print(f"Content Similarity:  {content_similarity.measure_similarity(q1, q2)}")
