# Hints for SPARQL Gen

Hints for SPARQL Gen generates SPARQL queries from natural-language questions using configurable LLM backends and ontology/schema context. It supports Hydra batch experiments and a FastAPI endpoint.

## Project Layout

Batch experiments use [Hydra](https://hydra.cc/) to compose a base configuration with an experiment preset. The base configuration defines shared selections, while each preset overrides the schema mode, prompt hints, and reasoning-path behavior.

* `main.py`: Hydra entry point for batch experiments.
* `config/config.yaml`: base selections and shared settings.
* `config/experiment/`: experiment presets and prompt-hint selections.
* `config/graphs/`: graph endpoints, question sets, and schema paths.
* `config/llm/`: LLM profiles.
* `config/prompt/`: graph-specific system prompts and hint blocks.
* `config/reasoning_path_prompt/`: reasoning-path prompts.
* `src/`: schema loading, context building, generation, validation, and LLM integrations.
* `api/`: FastAPI application.
* `slurm/`: local and cluster-oriented run scripts.
* `experiment/`, `outputs/`, and `logs/`: generated results and runtime logs.

## Installation

Use Python 3.11 or newer:

```powershell
python -m venv venv
venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Configure credentials locally before running. Never commit API keys or tokens. Use environment variables, an ignored local override, or the secret mechanism supported by the selected provider. Any credential-shaped values already present in tracked configuration should be revoked and replaced.

The included provider profiles read credentials from these environment variables:

| Provider | Environment variable |
|-|-|
| OpenAI | `OPENAI_API_KEY` |
| Cohere | `COHERE_API_KEY` |
| SAIA | `SAIA_API_KEY` |
| SCADS | `SCADS_API_KEY` |
| Hugging Face models | `HF_TOKEN` |

Only set the variables required by the profile you select. Public Hugging Face
models may work without `HF_TOKEN`, but private or gated models require it.

## Run With Docker

Build the API image from the repository root:

```powershell
docker build -t sparql-chain .
```

Start the API on port 8000:

```powershell
docker run --rm -p 8000:8000 `
  -e OPENAI_API_KEY `
  sparql-chain
```

Open `http://localhost:8000/docs` to view the API documentation. Pass any
additional provider credentials required by the selected LLM configuration with
`-e`; do not copy `.env` files or secrets into the image.

## Experiment Presets

|Preset|Schema mode|Contents|
|-|-|-|
|`base`|`small`|Default small schema and general constraints.|
|`c01`|`small`|Baseline graph-hint condition.|
|`c02_ia`|`ner`|NER schema context and CoT-style instruction.|
|`ca`|`full`|Full graph schema with domain/range information.|
|`ca_ia`|`full`|Full schema plus the CoT-style instruction.|
|`ca_ib`|`reasoning_path`|Reasoning-path context using the base prompt.|
|`ca_ic`|`reasoning_path`|Reasoning-path context using the graph-specific prompt.|
|`ca_ia_ic`|`reasoning_path`|Reasoning path, CoT instruction, and graph-specific prompt.|
|`ca_cb_ia_ic`|`reasoning_path`|The previous combination plus query, aggregation, function, and unit hints.|

The available schema modes are `full`, `small`, `ner`, `reasoning_path`, and `none`.

## Run A Specific Experiment

Run commands from the repository root. Hydra composes `config/config.yaml` with the selected file from `config/experiment/`:

```powershell
python main.py +experiment=c01
```

### Configure A Run

The main settings in `config/config.yaml` are:

* `graphs@_global_`: graph configuration name from `config/graphs/`.
* `prompt`: prompt configuration name from `config/prompt/`.
* `reasoning_path_prompt@_global_`: reasoning-path prompt name.
* `llm@llm`, `llm@evaluation_llm`, `llm@ner_llm`, `llm@schema_path_llm`, and `llm@reasoning_path_llm`: backend profiles from `config/llm/`.
* `output_dir`: destination for generated CSV files.
* `data.max_questions`: `-1` for all questions or a positive number for a smoke test.
* `evaluation.enabled`: enables or disables query evaluation. Blank `Query` cells are
  automatically skipped for gold-query comparison while generated-query syntax is
  still checked.

The graph config supplies `graph.name`, `graph.graph_uri`, `graph.validator_api`, the semicolon-separated `graph.questions_path`, and schema paths. Question CSVs must contain `Hand-Written Question` and `Query` columns.

Use command-line overrides for a short test without changing the YAML files:

```powershell
python main.py +experiment=ca_ia data.max_questions=5 output_dir=experiment/local_running/smoke_test
```

For a generation-only run, disable evaluation (including the evaluation LLM):

```powershell
python main.py +experiment=base evaluation.enabled=false
```

## Run All Experiments

The local matrix runner in `slurm/run_experiment_locallly.sh` currently runs `c01`, `c02_ia`, `ca_ia`, `ca_ib`, `ca_ic`, `ca_ia_ic`, and `ca_cb_ia_ic`. It writes timestamped logs under `logs/`.

Run it from the repository root:

```powershell
bash slurm/run_experiment_locallly.sh
```

If Bash is not available, run the listed `python main.py +experiment=...` commands individually or use a compatible shell.

## Outputs And Analysis

Generated CSVs contain the SPARQL query, validation results, timing, schema mode, model, and experiment metadata. To analyze one result directory:

```powershell
python performance_analysis.py --data-dir experiment/local_running/camels_gb --output-dir experiment/local_running/camels_gb
```

The analysis skips files without the required experiment metrics and writes reports beside the input files.

To combine the NFDI4Earth results for the Qwen, Mistral, and Command R model
runs into one workbook with one sheet per model, run:

```powershell
python analysis_output/csv2xlsl_results/export_nfdi4earth_experiments.py
```

The workbook is written to
`analysis_output/csv2xlsl_results/nfdi4earth_experiments.xlsx`. Conditions are
stacked in the order `c01`, `c02_ia`, `ca`, `ca_ia`, `ca_ib`, `ca_ic`,
`ca_ia_ic`, and `ca_cb_ia_ic`. Missing condition files are reported as
warnings and do not prevent the other results from being exported. Use
`--qwen-dir`, `--mistral-dir`, or `--command-r-dir` to point the exporter at
different result directories.


## Using The API

Start from the repository root:

```powershell
python api/main.py
```

The server listens on `http://localhost:8000`; interactive documentation is at `http://localhost:8000/docs`.

For API requests, set `include_all_prompt_hints: true` in `config/config.yaml`. This ensures that every non-empty prompt hint block is included regardless of the selected schema mode. Keep it `false` for batch experiments so experiments only use their configured hints.

Test one question at a time:

```powershell
$body = @{ question = "Which datasets are available?"; graph = "nfdi4earth"; schema_mode = "small"; execute = $false } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/generate -Method Post -ContentType "application/json" -Body $body
```

Change `question` and send the request again to test another question. `question`, `graph`, and `schema_mode` are required. `llm` is optional and defaults to the initialized backend. Set `execute` to `$true` to call the graph's `validator_api`; set it to `$false` to return only the generated query. Requests that change graph, schema mode, or LLM recompose the runtime configuration.

## Adding Components

### 1. Add A Graph

1. Add `config/graphs/<graph_name>.yaml` with `graph.name`, `graph.graph_uri`, `graph.validator_api`, `graph.questions_path`, and the relevant `full_schema_path`, `small_schema_path`, `lexicon_schema_path`, `ontology_path`, and `schema_paths_path` values.
2. Add `config/prompt/<graph_name>.yaml` with the graph's `system_message`, `tag_blocks`, and `user_message`. Keep hint names stable and descriptive.
3. Add the question CSV and schema artifacts referenced by the graph config. Verify the question columns are `Hand-Written Question` and `Query`.
4. Select both files in `config/config.yaml` (`graphs@_global_` and `prompt`) or override them on the command line.
5. Run a smoke test with `data.max_questions=1` and confirm schema loading, query generation, and validator access.

For graph-specific reasoning paths, add `config/reasoning_path_prompt/<graph_name>.yaml` and select it with `reasoning_path_prompt@_global_`. The profiling helper can create class, predicate, predicate-schema, and compact TXT outputs:

```powershell
python helper/hydroOntology/schema_profiling/run_schema_profiling.py --graph <graph_name>
```

### 2. Add An LLM

1. Add `config/llm/<profile_name>.yaml`. Set `llm` to the fully qualified backend class, plus `model_name`, provider endpoint settings, generation settings, and a credential reference.
2. If the provider is OpenAI-compatible, point the profile at `src/llm_backends.OpenAIBackend` or another existing compatible backend and set its base URL and model.
3. If the provider needs different request or response handling, implement a class in `src/llm_backends/` inheriting `BaseLLMBackend`, implementing `generate` and `build_prompt`. Export it from `src/llm_backends/__init__.py` for consistency.
4. Select the profile in `config/config.yaml`, or pass it as a Hydra override. The API exposes profiles based on the YAML filenames in `config/llm/`.
5. Test generation with one question before using the profile for evaluation, NER, schema-path, or reasoning-path roles.

Keep tokens and keys outside tracked YAML. For example, the Hugging Face-compatible profile should read `HF_TOKEN` at runtime rather than storing the token in the repository.

### 3. Add A Hint Block

1. Add a named block under `tag_blocks` in the selected `config/prompt/<graph_name>.yaml`.
2. Make the block narrowly scoped and explicit about the SPARQL behavior it changes. Keep the content compatible with the selected graph's schema.
3. Add the block name to `prompt_hints` in a new or existing `config/experiment/<experiment_name>.yaml`.
4. Run the experiment with one question and inspect the composed prompt or generated query to verify that the block is included.

Example:

```yaml
tag_blocks:
  measurement_constraint: |
    <measurement_constraint>
      - Use the measurement predicate only with its documented value type.
    </measurement_constraint>
```

```yaml
prompt_hints:
  - general_constraints
  - schema_general_constraints
  - measurement_constraint
```

`use_reasoning_path_hint: true` enables the reasoning-path context. The API can include all non-empty hint blocks when `include_all_prompt_hints` is enabled.



## Schema Profiling Pipeline

The schema profiling pipeline queries a graph endpoint and creates the schema artifacts (full, small, perdicate) used by experiments. Run it from the repository root:

```powershell
python helper/hydroOntology/schema_profiling/run_schema_profiling.py `
  --graph <graph_path> `
  --endpoint <sparql_endpoint>
```

For example:

```powershell
python helper/hydroOntology/schema_profiling/run_schema_profiling.py `
  --graph climateobservations/eobs-v31 `
  --endpoint https://obs.knowledgehub.nfdi4earth.de/sparql/
```

The `--graph` argument is required and must be the complete graph path,
including its prefix. The value is used unchanged in the SPARQL graph IRI.
Slashes are replaced with underscores only in generated filenames, so a graph
such as `climateobservations/eobs-v31` produces files containing
`climateobservations_eobs-v31` rather than creating nested directories.
The `--endpoint` argument is required. The script does not infer an endpoint
from the graph path.

URI-to-prefix mappings are loaded from
`helper/hydroOntology/schema_profiling/prefixes.yaml` by default. Provide a
different YAML mapping with `--prefixes-file`:

```powershell
python helper/hydroOntology/schema_profiling/run_schema_profiling.py `
  --graph climateobservations/eobs-v31 `
  --prefixes-file path/to/prefixes.yaml
```

The pipeline writes timestamped outputs to `helper/hydroOntology/schema_profiling/`. It generates class and predicate CSV files, a predicate-schema CSV file, and a predicate-schema TXT file.

Use an endpoint override or a different output directory when needed:

```powershell
python helper/hydroOntology/schema_profiling/run_schema_profiling.py --graph <graph_name> --endpoint <sparql_endpoint> --output-dir <output_directory>
```

After the pipeline finishes, update the graph configuration with the generated schema paths before running an experiment.

# Contact

- [Auriol Degbelo](https://sites.google.com/site/aurioldegbelo)

- Christina Naguib


# Acknowledgments

The work has been funded by the German Research Foundation through the Project [NFDI4Earth](https://www.nfdi4earth.de/). 
