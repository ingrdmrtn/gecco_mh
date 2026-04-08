[![arXiv Badge](https://img.shields.io/badge/arXiv-B31B1B?logo=arxiv&logoColor=fff&style=for-the-badge)](https://arxiv.org/abs/2502.00879)

# 🧠 GeCCo: Guided Generation of Computational Cognitive Models

Authors: [Milena Rmus](https://github.com/MilenaCCNlab) and [Akshay K. Jagadish](https://akjagadish.github.io/)

## 📘 Overview

Guided Generation of Computational Cognitive Models (GeCCo) is a pipeline for automated generation of computational cognitive models using large language models (LLMs).

Given the task instructions, participant data from cognitive tasks, model generation specs and a template function, GeCCo:

1. Prompts an LLM to generate candidate cognitive models as executable Python functions
2. Fits these models offline to the held-out participant data using maximum likelihood estimation (via scipy.optimize)
3. Evaluates the generated model using metrics such as Bayesian Information Criterion (BIC), and uses this performance metric to guide further model generation
4. Refines the generated models over multiple iterations based on structured feedback

![GeCCo Schematic](GeCCo.png)

## 🧩 Key Features

- 🧮 Task-agnostic design through configurable input data columns
- ⚙️ YAML configuration for tasks, data, LLM settings, and evaluation
- 🧱 Modular architecture (prompting, fitting, evaluation, feedback)
- 🤖 LLM-driven model generation as interpretable Python functions
- 📊 Automated fitting with multi-start L-BFGS-B optimization
- 📈 BIC/AIC tracking to identify the best models and iteration results
- 🔁 Iterative search loop with optional manual or LLM-generated feedback

## 📂 Repository Structure

```text
.
├── README.md
├── requirements.txt
├── config/
│   ├── decision_making.yaml
│   ├── schema.py
│   └── two_step.yaml
├── data/
│   ├── multi_attribute_decision_making.csv
│   ├── rlwm.csv
│   ├── standardize_data.py
│   └── two_step_data.csv
├── gecco/
│   ├── __init__.py
│   ├── run_gecco.py
│   ├── utils.py
│   ├── construct_feedback/
│   │   ├── __init__.py
│   │   └── feedback.py
│   ├── load_llms/
│   │   ├── __init__.py
│   │   ├── gpt_backend.py
│   │   ├── llama_backend.py
│   │   ├── model_loader.py
│   │   ├── qwen_backend.py
│   │   └── r1_backend.py
│   ├── offline_evaluation/
│   │   ├── __init__.py
│   │   ├── data_structures.py
│   │   ├── evaluation_functions.py
│   │   ├── fit_generated_models.py
│   │   └── utils.py
│   ├── prepare_data/
│   │   ├── __init__.py
│   │   ├── data2text.py
│   │   └── io.py
│   └── prompt_builder/
│       ├── __init__.py
│       ├── guardrails.py
│       └── prompt.py
├── results/
│   ├── multi_attribute_decision_making/
│   │   ├── bics/
│   │   └── models/
│   └── two_step_task/
│       ├── bics/
│       └── models/
└── scripts/
    ├── decision_making_demo.py
    └── two_step_demo.py
```

## 🚀 Installation

### Prerequisites

- Python ≥ 3.10
- pip or conda

### Install dependencies

```bash
git clone https://github.com/MilenaCCNlab/gecco.git
cd gecco
pip install -r requirements.txt
```

## 🧰 Requirements

See `requirements.txt` for full list. Core packages include:

- numpy, pandas, scipy
- torch, transformers
- pyyaml, pydantic
- openai (for OpenAI backend)

Optional (for local LLMs): vllm, accelerate

### API keys

GeCCo reads API keys from environment variables or a `.env` file in the project root (`.env` is gitignored). Create a `.env` file and add whichever keys you need:

```bash
# OpenAI (required if using provider: "openai")
OPENAI_API_KEY=your_openai_api_key_here

# Google Gemini (required if using provider: "gemini")
GEMINI_API_KEY=your_gemini_api_key_here

# KCL AI API (required if using provider: "kcl")
KCL_API_KEY=your_kcl_api_key_here

# HuggingFace (optional — increases rate limits and is required for gated models such as LLaMA)
HF_TOKEN=your_hf_token_here
```

Alternatively, export them in your shell before running:

```bash
export OPENAI_API_KEY=your_openai_api_key_here
export GEMINI_API_KEY=your_gemini_api_key_here
export KCL_API_KEY=your_kcl_api_key_here
export HF_TOKEN=your_hf_token_here
```

A HuggingFace token can be created at huggingface.co/settings/tokens. For gated models (e.g. LLaMA), you must also accept the model licence on the model's HuggingFace page.

### Using the Google Gemini API

GeCCo supports Google Gemini models via the `google-genai` Python SDK. To use Gemini:

1. **Get an API key** from [Google AI Studio](https://aistudio.google.com/apikey)
2. **Set the key** in your `.env` file or shell (see above)
3. **Install the SDK** (included in `requirements.txt`):

   ```bash
   pip install google-genai
   ```

4. **Configure your YAML** with `provider: "gemini"` and a supported model name:

```yaml
llm:
  provider: "gemini"
  base_model: "gemini-3-flash-preview"
  temperature: 0.2
  max_output_tokens: 2048
  models_per_iteration: 3
  include_feedback: true
```

### Using local LLMs

GeCCo supports running open-weight models locally via HuggingFace Transformers. The supported providers are:

| Provider value | Models | Gated? |
|----------------|--------|--------|
| `llama` | Meta LLaMA family | Yes (requires HF login + licence) |
| `qwen` | Alibaba Qwen family | No |
| `r1` | DeepSeek R1-Distilled | No |

**Recommended models by size:**

| `base_model` | Provider | Params | VRAM (bfloat16) | Min GPUs (40 GB A100) |
|--------------|----------|--------|-----------------|----------------------|
| `Qwen/Qwen2.5-1.5B-Instruct` | `qwen` | 1.5B | ~3 GB | 1 |
| `meta-llama/Llama-3.2-3B-Instruct` | `llama` | 3B | ~6 GB | 1 |
| `Qwen/Qwen2.5-7B-Instruct` | `qwen` | 7B | ~14 GB | 1 |
| `meta-llama/Meta-Llama-3.1-8B-Instruct` | `llama` | 8B | ~16 GB | 1 |
| `Qwen/Qwen2.5-14B-Instruct` | `qwen` | 14B | ~28 GB | 1 |
| `Qwen/Qwen2.5-32B-Instruct` | `qwen` | 32B | ~64 GB | 2 |
| `meta-llama/Meta-Llama-3.1-70B-Instruct` | `llama` | 70B | ~140 GB | 4 |
| `Qwen/Qwen2.5-72B-Instruct` | `qwen` | 72B | ~144 GB | 4 |
| `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` | `r1` | 70B | ~140 GB | 4 |

Larger models produce better code but require more VRAM. If your model exceeds a single GPU's memory, `device_map="auto"` will split it across multiple GPUs automatically — ensure your job requests enough GPUs (e.g. `--gres=gpu:4` in SLURM).

To use a local model, set `provider` and `base_model` in your YAML config:

```yaml
llm:
  provider: "llama"
  base_model: "meta-llama/Meta-Llama-3.1-70B-Instruct"
  temperature: 0.2
  max_output_tokens: 2048
```

Models are downloaded from the HuggingFace Hub on first use and loaded with `device_map="auto"` (automatically distributed across available GPUs). bfloat16 precision is used when CUDA is available. You will need:

- A GPU with sufficient VRAM for your chosen model (e.g. ~140 GB for a 70B model in bfloat16, or less with quantisation)
- `torch`, `transformers`, and `accelerate` installed (included in `requirements.txt`)
- For gated models (e.g. LLaMA), log in with `huggingface-cli login` and accept the model license on HuggingFace

**HPC users:** Models are cached in `~/.cache/huggingface/` by default, which may exceed home directory quotas. To use a different location (e.g. a scratch filesystem), set the `HF_HOME` environment variable in your shell or job script before running:

```bash
export HF_HOME=/scratch/$USER/huggingface
python scripts/two_step_demo.py --config config/two_step_local.yaml
```

We also have models stored for lab use, in which case you can set `HF_HOME` to that path:

```bash
export HF_HOME=/scratch/prj/bcn_neudec/huggingface
python scripts/two_step_demo.py --config config/two_step_local.yaml
```

Note: `HF_HOME` must be set as a shell environment variable — putting it in the `.env` file will not work, as HuggingFace reads it at import time before `python-dotenv` loads.

### Using a vLLM server (recommended for HPC)

Instead of loading a model in-process, you can serve it as a standalone API using [vLLM](https://docs.vllm.ai/) and have GeCCo query it over HTTP. This decouples model serving from model fitting, allowing you to:

- Reuse one LLM server across multiple GeCCo runs
- Run model fitting on CPU-only nodes while the LLM runs on GPU nodes
- Avoid reloading the model for each experiment

#### Step 0: Install vLLM

vLLM is not included in the base `requirements.txt` since it is only needed on the machine serving the model (not on nodes that just run GeCCo). Install it in its own environment to avoid dependency conflicts:

```bash
# Create a dedicated venv (recommended — vLLM pins specific torch/CUDA versions)
python -m venv ~/.venvs/vllm
source ~/.venvs/vllm/bin/activate

# Install vLLM (includes torch with CUDA support)
pip install vllm

# For gated models (e.g. LLaMA), log in to HuggingFace:
pip install huggingface-hub
huggingface-cli login
```

On HPC systems with module-managed CUDA, make sure a compatible CUDA toolkit is loaded before installing (e.g. `module load cuda/12.4`). vLLM requires CUDA 12.1+ at runtime.

**Quick check that it works:**

```bash
python -c "from vllm import LLM; print('vLLM installed successfully')"
```

#### Step 1: Launch the vLLM server

vLLM exposes an OpenAI-compatible HTTP API. You need to start this server on a machine with GPUs, then point GeCCo at it.

**Option A — Local / interactive (simplest):**

Open a terminal on your GPU machine and run:

**Important:** vLLM downloads models from the HuggingFace Hub. On HPC systems where home directories have limited quota, set `HF_HOME` to a scratch/shared location **before** launching the server:

```bash
# Point at the lab's shared model cache (or any path with enough space)
export HF_HOME=/scratch/prj/bcn_neudec/huggingface
```

If you already have models downloaded there, vLLM will use the cached copy. This must be set as a shell environment variable before the `python -m vllm...` command — it cannot be set in `.env`.

**Choosing `--tensor-parallel-size`:** This controls how many GPUs the model is split across. Set it based on your GPU memory and model size — the model weights (in bfloat16) must fit in the combined VRAM:

| GPU          | VRAM   | 14B model                  | 70B model                  |
| ------------ | ------ | -------------------------- | -------------------------- |
| A100         | 40 GB  | `--tensor-parallel-size 1` | `--tensor-parallel-size 4` |
| A100 (80 GB) | 80 GB  | `--tensor-parallel-size 1` | `--tensor-parallel-size 2` |
| H200         | 141 GB | `--tensor-parallel-size 1` | `--tensor-parallel-size 1` |
| B200         | 192 GB | `--tensor-parallel-size 1` | `--tensor-parallel-size 1` |

If omitted, vLLM defaults to 1 GPU. Using more GPUs than necessary still works but adds communication overhead.

```bash
# Small model for testing (~3 GB VRAM, no HF login required)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --port 8000

# LLaMA 70B — adjust --tensor-parallel-size for your GPU (see table above)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-70B-Instruct \
    --port 8000 \
    --tensor-parallel-size 4 \
    --trust-remote-code

# DeepSeek R1 distilled 70B
python -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
    --port 8000 \
    --tensor-parallel-size 4 \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code
```

Wait for the log line `Uvicorn running on http://0.0.0.0:8000` before proceeding.

**Option B — SLURM (HPC):**

```bash
# Usage: sbatch bash/launch_vllm_server.sh [MODEL] [PORT] [TP_SIZE]
sbatch bash/launch_vllm_server.sh meta-llama/Meta-Llama-3.1-70B-Instruct 8000 4
```

This submits a GPU job that starts vLLM and writes the server address to `$HOME/.vllm_env`. Check `logs/vllm-server-<jobid>.out` for startup progress.

**Verifying the server is ready:**

```bash
# Replace <hostname> with localhost (local) or the SLURM node hostname
curl http://<hostname>:8000/v1/models
```

You should see a JSON response listing the served model. If the connection is refused, the server is still loading — large models can take 2-5 minutes.

#### Step 2: Set the server URL

GeCCo reads the vLLM server address from the `VLLM_BASE_URL` environment variable.

If you used the SLURM script:

```bash
source $HOME/.vllm_env
```

Or set it directly:

```bash
export VLLM_BASE_URL=http://<hostname>:8000/v1
```

The URL **must** include the `/v1` suffix. If your vLLM server uses an API key (launched with `--api-key`), also set:

```bash
export VLLM_API_KEY=your_key_here
```

#### Step 3: Configure GeCCo to use vLLM

Set `provider: "vllm"` in your YAML config. The `base_model` must match the model name that vLLM is serving (by default, this is the HuggingFace model ID):

```yaml
llm:
  provider: "vllm"
  base_model: "meta-llama/Meta-Llama-3.1-70B-Instruct"
  temperature: 0.2
  max_output_tokens: 2048
```

See `config/two_step_vllm_example.yaml` for a complete example.

#### Step 4: Run GeCCo

```bash
python scripts/two_step_psychiatry_group.py --config two_step_vllm_example.yaml
```

GeCCo will connect to the running vLLM server instead of loading a model locally. The server must be running and ready before the script starts.

**Quick-start summary (local testing with a small model):**

```bash
# Terminal 1 — start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct --port 8000

# Terminal 2 — run GeCCo
export VLLM_BASE_URL=http://localhost:8000/v1
python scripts/two_step_psychiatry_group.py --config two_step_vllm_example.yaml
```

Note that model generation quality with small models (1.5B–3B) will be significantly lower than with larger models (14B+). Qwen models are ungated and can be downloaded without a HuggingFace account or licence agreement, making them the quickest option to get started.

### Distributed parallel search (multiple clients)

When model fitting is the bottleneck (e.g. fitting ~1000 subjects per model), you can run multiple GeCCo clients in parallel. Each client queries the same vLLM server for model generation but fits models independently. Clients share results via a JSON registry file on the shared filesystem, so each client's LLM feedback incorporates discoveries from all other clients.

```text
            vLLM Server (GPU node)
                   |  HTTP
        +----------+----------+
        |          |          |
    Client 0   Client 1   Client 2
    (CPU job)  (CPU job)  (CPU job)
        |          |          |
        +----+-----+-----+---+
             |
      Shared filesystem
      (shared_registry.json)
```

#### Step 1: Define client profiles in your config

Add a `clients:` section to your YAML config. Each profile can override LLM settings such as temperature and append extra instructions to the system prompt:

```yaml
clients:
  exploit:
    llm:
      temperature: 0.3
      system_prompt_suffix: |
        Focus on refining and improving the best-performing model.
        Make small, targeted changes to parameters and mechanisms.
  explore:
    llm:
      temperature: 0.9
      system_prompt_suffix: |
        Be creative and try fundamentally different model architectures.
        Consider cognitive mechanisms not yet explored.
      extra_guardrails:
        - "Each model must use at least one mechanism not present in any previous model."
  diverse:
    llm:
      temperature: 0.7
      models_per_iteration: 5
      system_prompt_suffix: |
        Ensure each proposed model uses a substantially different
        combination of cognitive mechanisms from any previous model.
  minimal:
    llm:
      temperature: 0.5
      system_prompt_suffix: |
        Propose simple models with few parameters. Prioritise parsimony.
      extra_guardrails:
        - "Each model should have at most 3 free parameters."
```

Available override fields per profile:

| Field | Effect |
| ------- | -------- |
| `temperature` | Direct override of `llm.temperature` |
| `models_per_iteration` | Direct override of `llm.models_per_iteration` |
| `system_prompt_suffix` | Appended to the base `llm.system_prompt` |
| `extra_guardrails` | Appended to the `llm.guardrails` list |
| Any other `llm.*` field | Direct override |

The `clients:` section is ignored by existing non-distributed scripts.

#### Step 2: Launch with the launcher script

`scripts/launch_distributed.py` reads profiles from your config, launches the vLLM server and client array, and wires up SLURM dependencies automatically:

```bash
# Launch vLLM + all profiles from config
python scripts/launch_distributed.py --config two_step_factors_distributed.yaml --launch-vllm

# vLLM already running — just launch clients, specifying the server URL
python scripts/launch_distributed.py --config two_step_factors_distributed.yaml \
    --vllm-url http://gpu-node:8000/v1

# Run only a subset of profiles
python scripts/launch_distributed.py --config two_step_factors_distributed.yaml --profiles exploit,minimal

# Add extra clients running the base config (no profile overrides)
python scripts/launch_distributed.py --config two_step_factors_distributed.yaml --extra-clients 2

# Preview commands without submitting
python scripts/launch_distributed.py --config two_step_factors_distributed.yaml --dry-run
```

**Conda environment**: If your project dependencies (e.g. `pydantic`, `scipy`) are installed in a specific conda environment, pass `--conda-env` so each SLURM client job activates it before running Python:

```bash
python scripts/launch_distributed.py --config two_step_factors_distributed.yaml \
    --vllm-url http://gpu-node:8000/v1 \
    --conda-env my_gecco_env
```

If running the shell script directly via `sbatch`, the conda environment is the 4th positional argument:

```bash
sbatch --array=0-3 bash/run_gecco_distributed.sh \
    two_step_factors_distributed.yaml \
    "exploit,explore,diverse,minimal" \
    "http://gpu-node:8000/v1" \
    "my_gecco_env"
```

For local testing without SLURM, run clients directly (ensure the correct conda environment is already active):

```bash
python scripts/run_gecco_distributed.py --config two_step_factors_distributed.yaml \
    --client-id 0 --client-profile exploit --vllm-url http://localhost:8000/v1
```

#### Step 3: Monitor progress

```bash
# One-shot status
python scripts/monitor_distributed.py --task two_step_factors

# Live dashboard (refreshes every 10s, Ctrl+C to exit)
python scripts/monitor_distributed.py --task two_step_factors --watch 10
```

#### Step 4 (optional): Streamlit dashboard (interactive, SSH tunnel)

An interactive web dashboard is available in `gecco-mh-dashboard/`.

Install dashboard dependencies:

```bash
pip install -r gecco-mh-dashboard/requirements.txt
```

On the remote machine (compute node), start Streamlit bound to all interfaces:

```bash
streamlit run gecco-mh-dashboard/app.py \
  --server.address 0.0.0.0 \
  --server.port 8501
```

From your local machine, create an SSH tunnel through the login node to the compute node:

```bash
ssh -N -L 8501:<compute-node>:8501 <user>@<hpc-login-node>
```

Example: `ssh -N -L 8501:node-42:8501 user@hpc.create.kcl.ac.uk`

Then open <http://127.0.0.1:8501> locally.

#### How coordination works

- Clients share results via `results/<task_name>/shared_registry.json` on the shared filesystem
- Before each iteration, clients merge cross-client history into their feedback — the LLM judge automatically sees the full model landscape from all clients
- The global best model is tracked across all clients
- File locking and atomic writes prevent corruption from concurrent access
- Output files include the client ID in their names (e.g. `iter0_client2_run0.txt`)

## ⚙️ Configuration

All experiment parameters are specified in YAML files under `config/`.

Key sections include:

- `task`: task description and modeling goal for the LLM
- `data`: dataset path/columns and narrative template used for prompting; note that the pipeline currently assumes a trial structure in the data
- `llm`: provider/base model and output constraints/guardrails
- `evaluation`: metric and optimizer options
- `feedback`: feedback mode between iterations (manual or llm-based)
- `loop`: number of model sampling iterations and independent runs

Example (`config/two_step.yaml`):

```yaml
task:
  name: "two_step_task"
  description: "Participants choose between spaceships and interact with aliens for rewards."
  goal: "Propose {models_per_iteration} cognitive models as Python functions: {model_names}"

data:
  path: "data/two_step_data.csv"
  id_column: "participant"
  input_columns: ["choice_1", "state", "choice_2", "reward"]
  data2text_function: "narrative"
  narrative_template: |
    The participant chose spaceship {choice_1}, traveled to planet {state},
    asked alien {choice_2}, and received {reward} coins.
  splits:
    prompt: "[1:3]"
    eval: "[4:14]"
    test: "[14:]"

llm:
  provider: "openai"
  base_model: "gpt-4"
  temperature: 0.2
  max_output_tokens: 2048
  models_per_iteration: 3
  include_feedback: true
  guardrails:
    - "Each model must be a standalone Python function"
    - "Function names: cognitive_model1, cognitive_model2, ..."
    - "Return negative log-likelihood of observed choices"
    - "Include clear docstrings with parameter bounds"

evaluation:
  metric: "bic"           # or "aic"
  optimizer: "L-BFGS-B"
  n_starts: 10

feedback:
  type: "manual"          # or "llm"

loop:
  max_iterations: 5
  max_independent_runs: 1

```

## 🎯 Usage

Quick start with demo scripts:

```bash
# Two-step decision task
python scripts/two_step_demo.py

# Multi-attribute decision making
python scripts/decision_making_demo.py
```

Programmatic usage:

```python
from config.schema import load_config
from gecco.prepare_data.io import load_data, split_by_participant
from gecco.prepare_data.data2text import get_data2text_function
from gecco.load_llms.model_loader import load_llm
from gecco.run_gecco import GeCCoModelSearch
from gecco.prompt_builder.prompt import PromptBuilderWrapper

# load config
cfg = load_config("config/two_step.yaml")

# load and prepare data
df = load_data(cfg.data.path, cfg.data.input_columns)
splits = split_by_participant(df, cfg.data.id_column, cfg.data.splits)

# get prompt and eval splits
df_prompt, df_eval = splits["prompt"], splits["eval"]

# convert data to narrative text
data2text = get_data2text_function(cfg.data.data2text_function)
data_text = data2text(
    df_prompt,
    id_col=cfg.data.id_column,
    template=cfg.data.narrative_template,
    value_mappings=getattr(cfg.data, "value_mappings", None),
)

# build prompt
prompt_builder = PromptBuilderWrapper(cfg, data_text)

# load llm
model, tokenizer = load_llm(cfg.llm.provider, cfg.llm.base_model)

# setup GeCCo
search = GeCCoModelSearch(model, tokenizer, cfg, df_eval, prompt_builder)

# run search
best_model, best_bic, best_params = search.run_n_shots(run_idx=0)

# print results: best model code, BIC, params
print("Best Model Code:\n", best_model)
print("Best BIC:", best_bic)
print("Best Parameters:", best_params)
```

## 🧪 How it works

1. Build a structured prompt with task description, example data (as a narrative), guardrails, and optional feedback
2. Generate multiple candidate models per iteration as Python functions
3. Extract function code, parameter names, and bounds
4. Fit each model to each participant via multi-start L-BFGS-B
5. Compute BIC/AIC and track the best model
6. Feed back guidance for the next iteration and repeat

## 📊 Output

After runs, results are saved under `results/<task_name>/`:

```text
results/two_step_task/
├── models/
│   ├── best_model.py
│   ├── iter0.py
│   ├── iter1.py
│   └── ...
└── bics/
    ├── iter0.json
    ├── iter1.json
    └── ...
```

Each JSON contains entries like:

```json
[
  {
    "function_name": "cognitive_model1",
    "metric_name": "BIC",
    "metric_value": 245.67,
    "param_names": ["alpha", "beta", "w"],
    "code_file": "results/two_step_task/models/iter0.py"
  }
]
```

## 🤝 Contributing

Contributions are welcome! Please open an issue or pull request.

## 📄 License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📚 Citation

If you use GeCCo in research, please cite:

```bibtex
@article{rmus2025generating,
  title={Generating Computational Cognitive Models using Large Language Models},
  author={Rmus, Milena and Jagadish, Akshay K. and Mathony, Marvin and Ludwig, Tobias and Schulz, Eric},
  journal={Advances in Neural Information Processing Systems},
  year={2025},
  url={https://arxiv.org/abs/2502.00879},
}
```

For questions or issues, please open a GitHub issue.
