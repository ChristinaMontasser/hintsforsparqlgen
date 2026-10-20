# Required columns for comparison analysis
REQUIRED_COLS = [
    # Identifiers
    "question_id", "question", "model_name", "schema_mode", "metric_query_type",
    
    # Quality metrics
    "sim_struct_combined_similarity", "metric_f1", "metric_jaccard", "metric_exact_count",
    
    # Validity metrics  
    "val_grammar_ok", "val_preparation_ok", "metric_error",
    
    # Efficiency metrics
    "execution_time_sec",
    
    # Metadata
    "run_timestamp"#, "question_type"  # optional: domain/category of question
]

OPTIONAL_NUMERIC_COLS = [
    "metric_answer_overlap_jaccard",
    "metric_answer_overlap_precision",
    "metric_answer_overlap_recall",
    "metric_answer_overlap_f1",
]

# analysis/compare_models.py
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
import warnings
warnings.filterwarnings('ignore')

def load_all_experiments(data_dir: str = "results/") -> pd.DataFrame:
    """Load all CSV files from directory and combine into single DataFrame."""
    
    all_dfs = []
    ignored_prefixes = ("ranking_",)
    ignored_names = {
        "multi_metric_report.csv",
        "metric_correlations.csv",
        "pareto_frontier.csv",
    }
    
    for filepath in Path(data_dir).glob("*.csv"):
        if filepath.name.startswith(ignored_prefixes) or filepath.name in ignored_names:
            continue

        try:
            df = pd.read_csv(filepath, sep=",", on_bad_lines='skip')

            if "experiment_name" not in df.columns:
                df["experiment_name"] = filepath.stem
            
            # Validate required columns
            missing = set(REQUIRED_COLS) - set(df.columns)
            if missing:
                print(f"⚠️  Skipping {filepath.name}: missing {missing}")
                continue
                
            all_dfs.append(df)
            print(f"✅ Loaded {filepath.name}: {len(df)} rows")
            
        except Exception as e:
            print(f"❌ Error loading {filepath}: {e}")
    
    if not all_dfs:
        raise ValueError("No valid experiment files found!")
    
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Clean numeric columns
    numeric_cols = [
        "metric_f1",
        "metric_jaccard",
        "execution_time_sec",
        "sim_struct_combined_similarity",
        *OPTIONAL_NUMERIC_COLS,
    ]
    for col in numeric_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors='coerce')

    if "metric_answer_overlap_f1" in combined.columns:
        overlap_count = combined["metric_answer_overlap_f1"].notna().sum()
        if overlap_count:
            print(f"🔎 Answer-overlap metric available for {overlap_count}/{len(combined)} rows")
    
    print(f"\n📊 Combined dataset: {len(combined)} rows × {len(combined.columns)} columns")
    print(f"🔍 Experiments: {combined['experiment_name'].nunique()} | Models: {combined['model_name'].nunique()} | Schema modes: {combined['schema_mode'].nunique()}")
    
    return combined

# analysis/multi_metric_comparator.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class MetricConfig:
    """Define all metrics with their properties for comparison."""
    
    METRICS = {
        # === SYNTAX VALIDITY ===
        "val_grammar_ok": {
            "group": "syntax",
            "name": "Grammar Validity",
            "higher_is_better": True,
            "type": "binary",
            "weight": 1.0,
            "query_type_filter": None  # Apply to all query types
        },
        "val_preparation_ok": {
            "group": "syntax",
            "name": "Preparation Validity", 
            "higher_is_better": True,
            "type": "binary",
            "weight": 1.0,
            "query_type_filter": None
        },
        "metric_error": {
            "group": "syntax",
            "name": "Error Rate",
            "higher_is_better": True, #Inverted
            "type": "binary",
            "weight": 1.0,  # Errors are important!
            "query_type_filter": None
        },
        
        # === STRUCTURE SIMILARITY ===
        "sim_struct_combined_similarity": {
            "group": "structure",
            "name": "Structural Similarity (Combined)",
            "higher_is_better": True,
            "type": "continuous",
            "weight": 1.0,  # High priority
            "query_type_filter": None
        },
        "sim_struct_string_similarity": {
            "group": "structure",
            "name": "String Similarity",
            "higher_is_better": True,
            "type": "continuous", 
            "weight": 1.0,
            "query_type_filter": None
        },
        "sim_struct_graph_similarity": {
            "group": "structure",
            "name": "Graph Similarity",
            "higher_is_better": True,
            "type": "continuous",
            "weight": 1.0,
            "query_type_filter": None
        },
        "sim_struct_semantic_similarity": {
            "group": "structure", 
            "name": "Semantic Similarity",
            "higher_is_better": True,
            "type": "continuous",
            "weight": 1.0,
            "query_type_filter": None
        },
        
        # === CONTENT/LANGUAGE SIMILARITY ===
        "sim_content": {
            "group": "content",
            "name": "Content Similarity",
            "higher_is_better": True,
            "type": "continuous",
            "weight": 1.0,
            "query_type_filter": None
        },
        "sim_lang_frequency_similarity": {
            "group": "language",
            "name": "Language Frequency Similarity",
            "higher_is_better": True,
            "type": "continuous",
            "weight": 1.0,
            "query_type_filter": None
        },
        "sim_lang_boolean_similarity": {
            "group": "language",
            "name": "Language Boolean Similarity", 
            "higher_is_better": True,
            "type": "continuous",
            "weight": 1.0,
            "query_type_filter": None
        },
        
        # === RESULT CORRECTNESS (Standard Queries) ===
        "metric_f1": {
            "group": "results",
            "name": "F1 Score",
            "higher_is_better": True,
            "type": "bounded",
            "weight": 1.0,  # Most important for results
            "query_type_filter": "standard"
        },
        "metric_precision": {
            "group": "results",
            "name": "Precision",
            "higher_is_better": True,
            "type": "bounded",
            "weight": 1.0,
            "query_type_filter": "standard"
        },
        "metric_recall": {
            "group": "results",
            "name": "Recall", 
            "higher_is_better": True,
            "type": "bounded",
            "weight": 1.0,
            "query_type_filter": "standard"
        },
        "metric_jaccard": {
            "group": "results",
            "name": "Jaccard Similarity",
            "higher_is_better": True,
            "type": "bounded",
            "weight": 1.0,
            "query_type_filter": "standard"
        },
        "metric_exact_match": {
            "group": "results",
            "name": "Exact Result Match",
            "higher_is_better": True,
            "type": "binary",
            "weight": 1.0,
            "query_type_filter": "standard"
        },

        # === ANSWER OVERLAP CORRECTNESS (Column-aware, Standard Queries) ===
        "metric_answer_overlap_f1": {
            "group": "answer_overlap",
            "name": "Answer Overlap F1",
            "higher_is_better": True,
            "type": "bounded",
            "weight": 1.0,
            "query_type_filter": "standard"
        },
        "metric_answer_overlap_precision": {
            "group": "answer_overlap",
            "name": "Answer Overlap Precision",
            "higher_is_better": True,
            "type": "bounded",
            "weight": 1.0,
            "query_type_filter": "standard"
        },
        "metric_answer_overlap_recall": {
            "group": "answer_overlap",
            "name": "Answer Overlap Recall",
            "higher_is_better": True,
            "type": "bounded",
            "weight": 1.0,
            "query_type_filter": "standard"
        },
        "metric_answer_overlap_jaccard": {
            "group": "answer_overlap",
            "name": "Answer Overlap Jaccard",
            "higher_is_better": True,
            "type": "bounded",
            "weight": 1.0,
            "query_type_filter": "standard"
        },
        
        # === RESULT CORRECTNESS (COUNT Queries) ===
        "metric_exact_count": {
            "group": "results",
            "name": "Count Accuracy",
            "higher_is_better": True,
            "type": "bounded",
            "weight": 1.0,  # Count accuracy is already in [0, 1]
            "query_type_filter": "count"
        },
        
        # === EFFICIENCY ===
        "execution_time_sec": {
            "group": "efficiency",
            "name": "Execution Time",
            "higher_is_better": False,
            "type": "continuous",
            "weight": 1.0,
            "query_type_filter": None
        }
    }
    
    # Group-level weights (1.0 aggregated scoring)
    GROUP_WEIGHTS = {
        "syntax": 1.0,      # Must be valid first
        "structure": 1.0,   # Query structure matters
        "content": 1.0,     # Language similarity less critical
        "language": 1.0,    # Least critical similarity
        "results": 1.0,     # End result is most important
        "answer_overlap": 1.0,  # Report separately; do not alter legacy overall score
        "efficiency": 1.0   # Nice to have, but secondary
    }


class MultiMetricComparator:
    """Compare configurations across all metric dimensions."""
    
    def __init__(self, df: pd.DataFrame, metric_config: MetricConfig = None):
        self.df = df.copy()
        self.config = metric_config or MetricConfig()
        self._preprocess_data()
    
    def _preprocess_data(self):
        """Clean and prepare data for comparison."""
        # Convert binary columns to numeric
        # TODO: it could accessed through the metric config
        for col in ["val_grammar_ok", "val_preparation_ok", "metric_exact_match", "metric_exact_count"]:
            if col in self.df.columns:
                normalized = self.df[col].astype(str).str.strip().str.lower().map({
                    "true": 1,
                    "false": 0,
                    "1": 1,
                    "0": 0,
                    "yes": 1,
                    "no": 0,
                })
                self.df[col] = pd.to_numeric(normalized, errors='coerce')
        
        # Convert error messages to an "error-free" score so higher = better.
        if "metric_error" in self.df.columns:
            error_free = 1 - self.df["metric_error"].notna().astype(int)
            self.df["metric_error"] = error_free.astype(float)
            self.df["metric_error_inverted"] = self.df["metric_error"]
        
        # Ensure numeric columns are float
        numeric_cols = [
            m for m, c in self.config.METRICS.items()
            if c["type"] in {"continuous", "bounded"}
        ]
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
    
    def normalize_metric(self, series: pd.Series, higher_is_better: bool) -> pd.Series:
        series = pd.to_numeric(series, errors='coerce')
        clean = series.dropna()
        if len(clean) < 2:
            # Instead of returning 0.5, return 1 if higher_is_better, else 0
            return pd.Series(1.0 if higher_is_better else 0.0, index=series.index)
        
        min_val, max_val = clean.min(), clean.max()
        if min_val == max_val:
            return pd.Series(1.0 if higher_is_better else 0.0, index=series.index)
        
        normalized = (series - min_val) / (max_val - min_val)
        return normalized if higher_is_better else (1 - normalized)
    
    
    def compute_group_scores(self, df_group: pd.DataFrame) -> Dict[str, float]:
        """Compute weighted score for each metric group."""
        group_scores = {}
        
        for group_name in self.config.GROUP_WEIGHTS.keys():
            # Get metrics in this group
            group_metrics = [
                (name, cfg) for name, cfg in self.config.METRICS.items() 
                if cfg["group"] == group_name and name in df_group.columns
            ]
            
            if not group_metrics:
                continue
            
            # Filter by query type if needed
            scores = []
            weights = []
            
            for metric_name, cfg in group_metrics:
                # Skip if query type doesn't match
                if cfg["query_type_filter"]:
                    query_types = df_group["metric_query_type"].unique()
                    if cfg["query_type_filter"] not in query_types:
                        # For mixed groups, only use matching rows
                        subset = df_group[df_group["metric_query_type"] == cfg["query_type_filter"]]
                    else:
                        subset = df_group
                else:
                    subset = df_group
                
                if subset[metric_name].dropna().empty:
                    continue
                
                metric_series = subset[metric_name]

                if cfg["type"] in {"binary", "bounded"}:
                    score_values = pd.to_numeric(metric_series, errors='coerce').dropna()
                    if len(score_values) > 0:
                        score_value = score_values.mean()
                        scores.append(score_value if cfg["higher_is_better"] else 1 - score_value)
                        weights.append(cfg["weight"])
                    continue

                # Normalize continuous metrics to make scales comparable.
                normalized = self.normalize_metric(
                    metric_series,
                    cfg["higher_is_better"]
                ).dropna()
                
                if len(normalized) > 0:
                    scores.append(normalized.mean())
                    weights.append(cfg["weight"])
            
            if scores and weights:
                # Weighted average within group
                group_score = np.average(scores, weights=weights)
                group_scores[group_name] = group_score
        
        return group_scores
    
    def compute_overall_score(self, df_group: pd.DataFrame) -> float:
        """Compute single overall score from all groups."""
        group_scores = self.compute_group_scores(df_group)
        
        if not group_scores:
            return np.nan
        
        # Weighted average across groups
        scores = []
        weights = []
        
        for group, score in group_scores.items():
            if group in self.config.GROUP_WEIGHTS:
                scores.append(score)
                weights.append(self.config.GROUP_WEIGHTS[group])
        
        if not scores:
            return np.nan
        
        return np.average(scores, weights=weights)
    
    def rank_configurations(self, by: str = "overall", 
                           group_by: str = "experiment_name",
                           query_type: str = None) -> pd.DataFrame:
        """
        Rank all configurations grouped by the requested column.
        
        by: "overall" | "syntax" | "structure" | "content" | "results" | "efficiency"
        group_by: column to compare, usually "experiment_name"
        query_type: None | "standard" | "count"
        """
        rankings = []
        
        if group_by not in self.df.columns:
            raise ValueError(f"Group-by column '{group_by}' not found in dataframe")

        # Group by experiment or another comparison axis
        for group_value, group in self.df.groupby(group_by):
            # Filter by query type if specified
            if query_type and "metric_query_type" in group.columns:
                group = group[group["metric_query_type"] == query_type]
            
            if len(group) < 5:  # Minimum samples
                continue
            
            if by == "overall":
                score = self.compute_overall_score(group)
            elif by in self.config.GROUP_WEIGHTS:
                group_scores = self.compute_group_scores(group)
                score = group_scores.get(by, np.nan)
            else:
                # Single metric ranking
                if by in group.columns:
                    metric_cfg = self.config.METRICS.get(by, {})
                    normalized = self.normalize_metric(
                        group[by].dropna(),
                        metric_cfg.get("higher_is_better", True)
                    )
                    score = normalized.mean() if len(normalized) > 0 else np.nan
                else:
                    score = np.nan
            
            rankings.append({
                group_by: group_value,
                "n_samples": len(group),
                "score": score,
                **{f"group_{g}": s for g, s in self.compute_group_scores(group).items()}
            })
        
        ranking_df = pd.DataFrame(rankings)
        
        if "score" in ranking_df.columns:
            ranking_df = ranking_df.dropna(subset=["score"])
            ranking_df["rank"] = ranking_df["score"].rank(ascending=False)
            ranking_df = ranking_df.sort_values("rank")
        
        return ranking_df
    
def plot_radar_comparison(df: pd.DataFrame, configs: List[tuple], 
                         metric_groups: List[str] = None,
                         output_path: str = "radar_comparison.png"):
    """
    Radar chart comparing multiple configurations across metric groups.
    
    configs: List of experiment names to compare. Tuples are still accepted for backwards compatibility.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    comparator = MultiMetricComparator(df)
    
    if metric_groups is None:
        metric_groups = list(comparator.config.GROUP_WEIGHTS.keys())
    
    # Compute scores for each config
    data = []
    for config in configs:
        if isinstance(config, tuple):
            experiment_name = config[0]
        else:
            experiment_name = config

        subset = df[df["experiment_name"] == experiment_name]
        if len(subset) < 5:
            continue
        
        group_scores = comparator.compute_group_scores(subset)
        scores = [group_scores.get(g, 0) for g in metric_groups]
        data.append((str(experiment_name), scores))
    
    if not data:
        print("⚠️  Insufficient data for radar plot")
        return
    
    # Radar chart setup
    angles = np.linspace(0, 2*np.pi, len(metric_groups), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(data)))
    
    for i, (label, scores) in enumerate(data):
        values = scores + scores[:1]  # Close the loop
        ax.plot(angles, values, 'o-', linewidth=2, label=label, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    # Labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_groups, size=11)
    ax.set_ylim(0, 1)
    ax.set_title("Multi-Metric Configuration Comparison\n(Higher = Better)", 
                pad=20, size=14)
    
    # Grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📈 Saved radar plot: {output_path}")


def plot_pareto_frontier(df: pd.DataFrame, 
                        x_metric: str = "execution_time_sec",
                        y_metric: str = "sim_struct_combined_similarity",
                        output_path: str = "pareto_frontier.png"):
    """
    Plot Pareto frontier: configurations that aren't dominated on both metrics.
    Lower x + Higher y = better (for time vs similarity example)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Aggregate by configuration
    agg = df.groupby(["experiment_name"]).agg({
        x_metric: "mean",
        y_metric: "mean",
        "metric_query_type": lambda x: x.mode()[0] if len(x.mode()) > 0 else "mixed"
    }).reset_index()
    
    # Remove NaN
    agg = agg.dropna(subset=[x_metric, y_metric])
    
    if len(agg) < 2:
        print("⚠️  Insufficient data for Pareto plot")
        return
    
    # Find Pareto frontier
    # A point is Pareto-optimal if no other point is better on both dimensions
    pareto_mask = np.ones(len(agg), dtype=bool)
    
    for i in range(len(agg)):
        for j in range(len(agg)):
            if i == j:
                continue
            # Check if j dominates i
            if (agg.loc[j, x_metric] <= agg.loc[i, x_metric] and 
                agg.loc[j, y_metric] >= agg.loc[i, y_metric] and
                (agg.loc[j, x_metric] < agg.loc[i, x_metric] or 
                 agg.loc[j, y_metric] > agg.loc[i, y_metric])):
                pareto_mask[i] = False
                break
    
    # Plot
    plt.figure(figsize=(10, 7))
    
    # Non-Pareto points
    plt.scatter(agg[~pareto_mask][x_metric], agg[~pareto_mask][y_metric], 
               c='lightgray', s=50, label='Dominated', alpha=0.6)
    
    # Pareto-optimal points
    plt.scatter(agg[pareto_mask][x_metric], agg[pareto_mask][y_metric], 
               c='red', s=100, label='Pareto Optimal', edgecolors='black', zorder=5)
    
    # Label Pareto points
    for _, row in agg[pareto_mask].iterrows():
        plt.annotate(f"{row['experiment_name']}", 
                    (row[x_metric], row[y_metric]), 
                    textcoords="offset points", xytext=(5,5), ha='left', fontsize=8)
    
    plt.xlabel(f"{x_metric} {'↓' if 'time' in x_metric.lower() else '↑'}")
    plt.ylabel(f"{y_metric} ↑")
    plt.title(f"Pareto Frontier: {y_metric} vs {x_metric}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f" Saved Pareto plot: {output_path}")


def plot_correlation_matrix(df: pd.DataFrame, 
                           metrics: List[str] = None,
                           output_path: str = "metric_correlations.png"):
    """Plot correlation heatmap between all metrics."""
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    if metrics is None:
        # Pick representative metrics from each group
        metrics = [
            #"val_grammar_ok",
            "sim_struct_combined_similarity", 
            "sim_content",
            "metric_f1",
            "metric_answer_overlap_f1",
            "execution_time_sec"
        ]
    
    # Filter to available metrics
    available = [m for m in metrics if m in df.columns]
    if len(available) < 2:
        return
    
    # Compute correlations (drop NaN pairwise)
    corr_data = df[available].dropna()
    corr = corr_data.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
               square=True, linewidths=0.5)
    plt.title("Metric Correlation Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f" Saved correlation matrix: {output_path}")
    
    # Print key insights
    print("\n🔍 Notable Correlations:")
    for i, m1 in enumerate(available):
        for m2 in available[i+1:]:
            val = corr.loc[m1, m2]
            if abs(val) > 0.5:
                direction = "positively" if val > 0 else "negatively"
                print(f"  • {m1} & {m2}: {direction} correlated (r={val:.2f})")


def multi_metric_statistical_test(df: pd.DataFrame, 
                                  group_by: str = "experiment_name",
                                  stratify_by: Dict[str, str] = None,
                                  alpha: float = 0.05,
                                  correction: str = "bonferroni") -> Dict:
    """
    Run statistical tests across ALL metrics simultaneously.
    Returns dict with results per metric + overall summary.
    """
    from scipy import stats
    from statsmodels.stats.multitest import multipletests
    
    results = {
        "by_metric": {},
        "summary": {
            "total_tests": 0,
            "significant_count": 0,
            "significant_metrics": []
        }
    }
    
    comparator = MultiMetricComparator(df)
    prepared_df = comparator.df
    
    # Test each metric
    for metric_name, cfg in comparator.config.METRICS.items():
        if metric_name not in prepared_df.columns:
            continue
        
        # Filter by query type if needed
        test_df = prepared_df.copy()
        if cfg["query_type_filter"] and "metric_query_type" in test_df.columns:
            test_df = test_df[test_df["metric_query_type"] == cfg["query_type_filter"]]
        
        # Apply additional stratification
        if stratify_by:
            for col, val in stratify_by.items():
                if col in test_df.columns:
                    test_df = test_df[test_df[col] == val]
        
        # Need minimum samples per group
        grouped = test_df.groupby(group_by)[metric_name]
        groups = [g.dropna().values for _, g in grouped]
        group_names = [name for name, _ in grouped]
        
        if len([g for g in groups if len(g) >= 3]) < 2:
            continue  # Skip if insufficient data
        
        results["summary"]["total_tests"] += 1
        
        # Choose test based on data properties
        clean_groups = [g for g in groups if len(g) >= 3]
        if not clean_groups:
            continue
            
        # Normality check (Shapiro-Wilk)
        normality_pvals = [stats.shapiro(g)[1] for g in clean_groups if len(g) <= 5000]
        is_normal = all(p > 0.05 for p in normality_pvals) if normality_pvals else False
        
        # Equal variance check
        equal_var = stats.levene(*clean_groups)[1] > 0.05 if len(clean_groups) >= 2 else True
        
        # Run omnibus test
        if is_normal and equal_var and len(clean_groups) >= 2:
            stat, p_omnibus = stats.f_oneway(*clean_groups)
            test_name = "ANOVA"
        else:
            stat, p_omnibus = stats.kruskal(*clean_groups)
            test_name = "Kruskal-Wallis"
        
        # Effect size
        n_total = sum(len(g) for g in clean_groups)
        n_groups = len(clean_groups)
        if test_name == "ANOVA":
            # Eta-squared
            all_vals = np.concatenate(clean_groups)
            grand_mean = np.mean(all_vals)
            ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in clean_groups)
            ss_total = sum((x - grand_mean)**2 for g in clean_groups for x in g)
            effect_size = ss_between / ss_total if ss_total > 0 else 0
        else:
            # Epsilon-squared approximation
            h = stat
            effect_size = max(0, (h - n_groups + 1) / (n_total - n_groups))
        
        # Store result
        results["by_metric"][metric_name] = {
            "metric_name": cfg["name"],
            "group": cfg["group"],
            "test": test_name,
            "statistic": float(stat),
            "p_omnibus": float(p_omnibus),
            "effect_size": float(effect_size),
            "group_means": {name: float(np.mean(g)) if len(g) > 0 else np.nan 
                          for name, g in zip(group_names, groups)},
            "group_counts": {name: len(g) for name, g in zip(group_names, groups)},
            "significant_uncorrected": p_omnibus < alpha
        }
        
        if p_omnibus < alpha:
            results["summary"]["significant_count"] += 1
            results["summary"]["significant_metrics"].append(metric_name)
    
    # Multiple comparison correction across all metrics
    if results["by_metric"] and correction:
        p_values = [r["p_omnibus"] for r in results["by_metric"].values()]
        metric_names = list(results["by_metric"].keys())
        
        if correction == "bonferroni":
            _, p_adj, _, _ = multipletests(p_values, alpha=alpha, method='bonferroni')
        elif correction == "fdr":
            _, p_adj, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
        else:
            p_adj = p_values
        
        for name, adj_p in zip(metric_names, p_adj):
            results["by_metric"][name]["p_corrected"] = float(adj_p)
            results["by_metric"][name]["significant_corrected"] = adj_p < alpha
    
    return results


def compute_llm_evaluation_by_experiment(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-experiment LLM equivalence counts and normalized rates."""
    if "metric_llm_equivalence_decision" not in df.columns:
        return pd.DataFrame()

    llm_per_experiment = (
        df.assign(
            _llm_decision=df["metric_llm_equivalence_decision"]
            .fillna("missing")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        .groupby("experiment_name")["_llm_decision"]
        .value_counts()
        .unstack(fill_value=0)
    )

    for decision_col in ["yes", "no", "missing"]:
        if decision_col not in llm_per_experiment.columns:
            llm_per_experiment[decision_col] = 0

    llm_per_experiment = llm_per_experiment[["yes", "no", "missing"]].copy()
    llm_per_experiment["total"] = llm_per_experiment.sum(axis=1)
    llm_per_experiment["decided"] = llm_per_experiment["yes"] + llm_per_experiment["no"]
    llm_per_experiment["yes_rate_total"] = (
        llm_per_experiment["yes"] / llm_per_experiment["total"].replace(0, np.nan)
    ).fillna(0.0)
    llm_per_experiment["no_rate_total"] = (
        llm_per_experiment["no"] / llm_per_experiment["total"].replace(0, np.nan)
    ).fillna(0.0)
    llm_per_experiment["yes_rate_decided"] = (
        llm_per_experiment["yes"] / llm_per_experiment["decided"].replace(0, np.nan)
    ).fillna(0.0)

    llm_per_experiment = llm_per_experiment.sort_values(
        by=["yes_rate_total", "yes_rate_decided", "yes", "no"],
        ascending=[False, False, False, True],
    )

    return llm_per_experiment


def generate_multi_metric_report(df: pd.DataFrame, output_path: str = "multi_metric_report.md"):
    """Generate comprehensive markdown report covering all dimensions."""
    
    comparator = MultiMetricComparator(df)
    report = []
    
    # Header
    report.append("#  Multi-Dimensional Experiment Comparison Report\n")
    report.append(f"*Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*\n")
    
    # Dataset overview
    report.append("##  Dataset Overview\n")
    report.append(f"- **Total evaluations**: {len(df):,}")
    report.append(f"- **Experiments**: {df['experiment_name'].nunique()} unique")
    report.append(f"- **Models**: {', '.join(df['model_name'].unique())}")
    report.append(f"- **Schema modes**: {', '.join(df['schema_mode'].unique())}")
    report.append(f"- **Query types**: {', '.join(df['metric_query_type'].dropna().unique())}\n")
    if "metric_answer_overlap_f1" in df.columns:
        overlap_count = df["metric_answer_overlap_f1"].notna().sum()
        report.append(f"- **Answer-overlap metric rows**: {overlap_count:,}/{len(df):,}")
        report.append("- **Note**: answer-overlap is reported separately and does not change the legacy weighted overall score.\n")

    if "metric_llm_equivalence_decision" in df.columns:
        llm_counts = (
            df["metric_llm_equivalence_decision"]
            .fillna("missing")
            .astype(str)
            .str.strip()
            .str.lower()
            .value_counts()
        )
        yes_count = int(llm_counts.get("yes", 0))
        no_count = int(llm_counts.get("no", 0))
        other_count = int(len(df) - yes_count - no_count)
        report.append(f"- **LLM evaluation decisions**: yes={yes_count:,}, no={no_count:,}, other={other_count:,}\n")

        llm_per_experiment = compute_llm_evaluation_by_experiment(df)

        report.append("## LLM Evaluation Decisions by Experiment\n")
        report.append("*Normalized values are shown as rates in [0, 1]. `yes_rate_decided` uses only decided rows (`yes` + `no`).*\n")
        report.append("| Experiment | Yes | No | Other | yes_rate_total | no_rate_total | yes_rate_decided |")
        report.append("|------------|-----|----|-------|----------------|---------------|------------------|")
        for exp_name, row in llm_per_experiment.iterrows():
            report.append(
                f"| {exp_name} | {int(row['yes'])} | {int(row['no'])} | {int(row['missing'])} | "
                f"{row['yes_rate_total']:.3f} | {row['no_rate_total']:.3f} | {row['yes_rate_decided']:.3f} |"
            )
        report.append("")
    
    # Overall rankings
    report.append("## 🏆 Overall Rankings (Weighted Multi-Metric Score)\n")
    overall_ranking = comparator.rank_configurations(by="overall")
    
    if not overall_ranking.empty:
        report.append("| Rank | Experiment | Score | Samples |")
        report.append("|------|------------|-------|---------|")
        for _, row in overall_ranking.head(10).iterrows():
            report.append(f"| {int(row['rank'])} | {row['experiment_name']} | {row['score']:.3f} | {int(row['n_samples'])} |")
        report.append("")
    
    # Rankings by dimension
    for dimension in ["syntax", "structure", "results", "answer_overlap", "efficiency"]:
        report.append(f"## Rankings by {dimension.upper()} Dimension\n")
        dim_ranking = comparator.rank_configurations(by=dimension)
        
        if not dim_ranking.empty:
            report.append(f"*Higher score = better performance on {dimension} metrics*\n")
            report.append("| Rank | Experiment | Score |")
            report.append("|------|------------|-------|")
            for _, row in dim_ranking.head(5).iterrows():
                report.append(f"| {int(row['rank'])} | {row['experiment_name']} | {row['score']:.3f} |")
            report.append("")
    
    # Statistical findings
    report.append("## 🔬 Statistical Significance Testing\n")
    stats_results = multi_metric_statistical_test(df)
    
    if stats_results["summary"]["significant_count"] > 0:
        report.append(f" Found {stats_results['summary']['significant_count']} metrics with significant differences across experiments (p < 0.05)\n")
        
        for metric_name, result in stats_results["by_metric"].items():
            if result.get("significant_corrected", result.get("significant_uncorrected", False)):
                cfg = comparator.config.METRICS[metric_name]
                report.append(f"- **{cfg['name']}** ({cfg['group']}): p={result['p_corrected'] if 'p_corrected' in result else result['p_omnibus']:.4f}, η²={result['effect_size']:.3f}")
                # Show best group
                if result["group_means"]:
                    best = max(result["group_means"], key=result["group_means"].get)
                    report.append(f"  → Best: `{best}` ({result['group_means'][best]:.3f})")
                report.append("")
    else:
        report.append(" No metrics showed statistically significant differences across experiments at α=0.05 (after multiple comparison correction)\n")
    
    # Key insights by use case
    report.append("## Recommendations by Use Case\n")
    
    # Best for accuracy (results-focused)
    results_ranking = comparator.rank_configurations(by="results", query_type="standard")
    if not results_ranking.empty:
        best = results_ranking.iloc[0]
        report.append(f"### Highest Result Accuracy (Standard Queries)\n")
        report.append(f"**Winner**: `{best['experiment_name']}`\n")
        report.append(f"- Expected F1: ~{df[(df['experiment_name']==best['experiment_name']) & (df['metric_query_type']=='standard')]['metric_f1'].mean():.3f}\n")

    # Best for answer overlap accuracy
    if "metric_answer_overlap_f1" in df.columns and df["metric_answer_overlap_f1"].notna().any():
        answer_overlap_ranking = comparator.rank_configurations(by="answer_overlap", query_type="standard")
        if not answer_overlap_ranking.empty:
            best = answer_overlap_ranking.iloc[0]
            best_rows = df[
                (df["experiment_name"] == best["experiment_name"])
                & (df["metric_query_type"] == "standard")
            ]
            report.append(f"### Highest Answer Overlap Accuracy (Standard Queries)\n")
            report.append(f"**Winner**: `{best['experiment_name']}`\n")
            report.append(f"- Expected answer-overlap F1: ~{best_rows['metric_answer_overlap_f1'].mean():.3f}\n")
            report.append("- This metric compares shared answer variables when result tables have different columns.\n")
    
    # Best for speed
    efficiency_ranking = comparator.rank_configurations(by="efficiency")
    if not efficiency_ranking.empty:
        best = efficiency_ranking.iloc[0]
        report.append(f"### Fastest Execution\n")
        report.append(f"**Winner**: `{best['experiment_name']}`\n")
        report.append(f"- Avg time: ~{df[df['experiment_name']==best['experiment_name']]['execution_time_sec'].mean():.2f}s\n")
    
    # Best for syntax validity
    syntax_ranking = comparator.rank_configurations(by="syntax")
    if not syntax_ranking.empty:
        best = syntax_ranking.iloc[0]
        report.append(f"###  Most Syntactically Valid Queries\n")
        report.append(f"**Winner**: `{best['experiment_name']}`\n")
        grammar_rate = df[df['experiment_name']==best['experiment_name']]['val_grammar_ok'].mean()
        report.append(f"- Grammar validity: {grammar_rate:.1%}\n")
    
    # Balanced recommendation
    report.append(f"### Best Balanced Configuration\n")
    balanced = overall_ranking.iloc[0] if not overall_ranking.empty else None
    if balanced is not None:
        report.append(f"**Recommended**: `{balanced['experiment_name']}`\n")
        report.append("- Good tradeoff across validity, similarity, correctness, and efficiency\n")
    
    # Save report
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    
    print(f" Report saved: {output_path}")
    return "\n".join(report)


# main_multi_metric_analysis.py
def main():
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Multi-metric experiment comparison")
    parser.add_argument("--data-dir", default="results/hydroOntology/camels_gb", help="Directory with CSV files")
    parser.add_argument("--output-dir", default="results/hydroOntology/camels_gb", help="Output directory")
    parser.add_argument("--query-type", choices=[None, "standard", "count"], default=None)
    parser.add_argument("--group-by", default="experiment_name", help="Column to compare across, defaults to experiment_name")
    args = parser.parse_args()
    print(args.output_dir)
    print(args.data_dir)
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(" Loading experiment data...")
    df = load_all_experiments(args.data_dir)  # From earlier code
    
    if args.query_type:
        df = df[df["metric_query_type"] == args.query_type]
        print(f" Filtered to {args.query_type} queries: {len(df)} rows")
    
    # Initialize comparator
    comparator = MultiMetricComparator(df)
    if args.group_by not in df.columns:
        raise ValueError(f"Requested group-by column '{args.group_by}' is not present in the loaded data")
    
    # Generate rankings
    print("\n Computing rankings...")
    for dimension in ["overall", "syntax", "structure", "results", "answer_overlap", "efficiency"]:
        ranking = comparator.rank_configurations(by=dimension, group_by=args.group_by, query_type=args.query_type)
        if not ranking.empty:
            ranking.to_csv(output_dir / f"ranking_{dimension}.csv", index=False)
            print(f"  {dimension}: Top = {ranking.iloc[0][args.group_by]}")

    # Export per-experiment normalized LLM-evaluation metrics.
    llm_eval = compute_llm_evaluation_by_experiment(df)
    if not llm_eval.empty:
        llm_eval.to_csv(output_dir / "llm_evaluation_by_experiment.csv", index=True, index_label="experiment_name")
        print("  llm_evaluation: Saved llm_evaluation_by_experiment.csv")
    
    # Generate visualizations
    print("\n Generating visualizations...")
    
    # Radar chart for top configurations
    top_configs = comparator.rank_configurations(by="overall", group_by=args.group_by).head(4)[[args.group_by]]
    configs_to_plot = top_configs[args.group_by].tolist()
    
    plot_radar_comparison(
        df, configs_to_plot,
        output_path=str(output_dir / "radar_top_configs.png")
    )
    
    # # Pareto: Accuracy vs Speed
    # plot_pareto_frontier(
    #     df,
    #     x_metric="execution_time_sec",
    #     y_metric="sim_struct_combined_similarity",
    #     output_path=str(output_dir / "pareto_accuracy_vs_speed.png")
    # )
    
    # Correlation matrix
    plot_correlation_matrix(
        df,
        output_path=str(output_dir / "metric_correlations.png")
    )
    
    # Generate report
    print("\n Generating comprehensive report...")
    generate_multi_metric_report(df, output_path=str(output_dir / "multi_metric_report.md"))
    
    # Print quick summary
    print("\n" + "="*60)
    print(" ANALYSIS COMPLETE")
    print("="*60)
    print(f" Outputs saved to: {output_dir}/")
    print(" Start with: multi_metric_report.md")
    print(" Key plots:")
    print("   • radar_top_configs.png - Multi-dimension comparison")
    print("   • pareto_accuracy_vs_speed.png - Tradeoff analysis")
    print("   • metric_correlations.png - Metric relationships")
    print("\n Pro tip: Filter by query_type for more focused insights:")
    print("   python main_multi_metric_analysis.py --query-type standard")

if __name__ == "__main__":
    main()
