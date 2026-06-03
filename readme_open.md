[![arXiv Badge](https://img.shields.io/badge/arXiv-B31B1B?logo=arxiv&logoColor=fff&style=for-the-badge)](https://arxiv.org/abs/2502.00879)

# 🧠 GeCCo (OpenRouter edition): Guided Generation of Computational Cognitive Models

Authors: [Milena Rmus](https://github.com/MilenaCCNlab) and [Akshay K. Jagadish](https://akjagadish.github.io/)

> This README is a slimmed-down variant of the main [README.md](README.md) that
> drops vLLM and local-GPU model serving entirely, and instead routes **all
> LLM calls through [OpenRouter](https://openrouter.ai)** using a single
> `OPENROUTER_API_KEY` loaded from a `.env` file. Use this when you don't
> want to manage a model server (no GPU node, no `--launch-vllm`, no
> HuggingFace cache) and just want to call hosted models.

## 📘 Overview

Guided Generation of Computational Cognitive Models (GeCCo) is a pipeline for automated generation of computational cognitive models using large language models (LLMs).

Given the task instructions, participant data from cognitive tasks, model generation specs and a template function, GeCCo:

1. Prompts an LLM to generate candidate cognitive models as executable Python functions
2. Fits these models offline to the held-out participant data using maximum likelihood estimation (via scipy.optimize)
3. Evaluates the generated model using metrics such as Bayesian Information Criterion (BIC), and uses this performance metric to guide further model generation
4. Refines the generated models over multiple iterations based on structured feedback

![GeCCo Schematic](GeCCo.png)

## 🚀 Installation

### Prerequisites

- Python ≥ 3.10
- pip or conda

### Install dependencies

```bash
make sure you make a conda envioremtn and then do the install 
pip install -r requirements.txt
pip install -r gecco-mh-dashboard/requirements.txt
```

You do **not** need `vllm`, `torch` with CUDA, `accelerate`, or any GPU. The
OpenRouter backend only needs the `openai` and `python-dotenv` packages
(both already in `requirements.txt`).

## 🔑 OpenRouter setup

GeCCo's OpenRouter backend ([gecco/load_llms/openrouter_backend.py](gecco/load_llms/openrouter_backend.py)) reads two environment variables:

| Variable | Required? | Default |
| -------- | --------- | ------- |
| `OPENROUTER_API_KEY` | yes | — |
| `OPENROUTER_BASE_URL` | no | `https://openrouter.ai/api/v1` |

The backend calls `load_dotenv()` at import time, so the key can live in a
`.env` file rather than your shell environment.

### Option 1 — `.env` in the project root

Create a `.env` file in the directory you run GeCCo from (typically the
repo root):

```bash
OPENROUTER_API_KEY=sk-or-...your-key-here...
```

Then run any script from that same directory.

### Quick sanity check

```bash
cd /scratch/prj/bcn_neudec/gecco_mh   # or wherever your .env lives
python -c "from gecco.load_llms.openrouter_backend import load_openrouter; \
           c = load_openrouter('openai/gpt-4o-mini'); \
           print(c.models.list().data[0].id)"
```

If the key is wired up correctly you'll see a model id printed. If you
get `OPENROUTER_API_KEY not found`, your cwd doesn't contain a `.env`
with that key.

## ⚙️ Configuring a YAML to use OpenRouter

Set `provider: "openrouter"` in the `llm:` section of your config and
pick any [model offered by OpenRouter](https://openrouter.ai/models) as
`base_model`. The model id is whatever OpenRouter advertises — typically
`<vendor>/<model-name>`.

Minimal example:

```yaml
llm:
  provider: "openrouter"
  base_model: "openai/gpt-4o-mini"
  temperature: 0.2
  max_tokens: 4096
  models_per_iteration: 3
  include_feedback: true
  guardrails:
    - "Each model must be a standalone Python function"
    - "Function names: cognitive_model1, cognitive_model2, ..."
    - "Return negative log-likelihood of observed choices"
    - "Include clear docstrings with parameter bounds"
```

Several full example configs in this repo already use OpenRouter and can
be copied / adapted:

- [config/two_step_factors_gpt54nano.yaml](config/two_step_factors_gpt54nano.yaml) — `openai/gpt-5.4-nano`
- [config/two_step_factors_gemini31_flash_lite.yaml](config/two_step_factors_gemini31_flash_lite.yaml) — `google/gemini-3-flash-lite`
- [config/two_step_factors_glm5.yaml](config/two_step_factors_glm5.yaml) — `z-ai/glm-5`
- [config/two_step_factors_minimax_m2.yaml](config/two_step_factors_minimax_m2.yaml) — `minimax/m2`
- [config/two_step_factors_step35.yaml](config/two_step_factors_step35.yaml)
- [config/two_step_factors_gemma4.yaml](config/two_step_factors_gemma4.yaml)
- [config/two_step_factors_nemotron.yaml](config/two_step_factors_nemotron.yaml)
- [config/two_step_factors_qwen36plus_judge.yaml](config/two_step_factors_qwen36plus_judge.yaml)

To swap to a different OpenRouter-hosted model, the only field that has
to change is `base_model`.


## 🧑‍🤝‍🧑 Distributed parallel search (multiple clients, OpenRouter)

For larger searches you can fan out multiple clients that all coordinate
through a shared registry on the filesystem. Because every client just
hits OpenRouter directly, there is no model server to launch — the
diagram simplifies to:

```text
            OpenRouter API (HTTPS)
                   |
        +----------+----------+
        |          |          |
    Client 0   Client 1   Client 2
    (CPU job)  (CPU job)  (CPU job)
        |          |          |
        +----+-----+-----+----+
             |
      Shared filesystem
      (shared_registry.json)
```

### Step 1 — Define client profiles in your YAML

Same `clients:` block as the main README. Each profile overrides parts
of the `llm:` section (temperature, system prompt suffix, extra
guardrails, etc.). See [config/two_step_factors_gpt54nano.yaml](config/two_step_factors_gpt54nano.yaml) for a worked example with
`exploit`, `explore`, `minimal`, `hybrid`, `complex`, `bayesian`, and
`latent_state_inference` profiles.

### Step 2 — Launch via the launcher script

[scripts/launch_distributed.py](scripts/launch_distributed.py) detects the provider from the YAML's `llm.provider` field. 

Run:

```bash
cd /scratch/prj/bcn_neudec/gecco_mh
python scripts/launch_distributed.py \
    --config two_step_factors_gpt54nano.yaml \
    --conda-env gecco_mh                    # name of your conda env
```

Useful flags:

| Flag | Meaning |
| ---- | ------- |
| `--profiles a,b,c` | Run only a subset of the `clients:` profiles |
| `--extra-clients N` | Add N clients running the base config (no profile) |
| `--conda-env <name>` | Conda env each SLURM client should activate |
| `--partition <name>` | SLURM partition override |
| `--cpus-per-task N` | CPUs per client |
| `--dry-run` | Print sbatch commands without submitting |

**Important — make sure the SLURM job sees `OPENROUTER_API_KEY`.** SLURM
jobs do not always inherit your interactive shell's environment, and
`load_dotenv()` only reads `.env` from the job's cwd. Two reliable
options:


If clients fail with `OPENROUTER_API_KEY not found in environment`, that's the symptom — the SLURM job didn't pick up your `.env`.



### Step Monitor — Streamlit dashboard

The dashboard reads result files only — it does **not** call any LLM,
so no API key is needed for this step.



# On the compute node, from the project root so it can find results/:
cd /scratch/prj/bcn_neudec/gecco_mh
streamlit run gecco-mh-dashboard/app.py \
    --server.address 0.0.0.0 \
    --server.port 8501
```

From your laptop, open an SSH tunnel through the login node:

```bash
ssh -N -L 8501:<compute-node>:8501 <user>@hpc.create.kcl.ac.uk
```
Nothing will happen here that is ok you then open a new broswer window and 
Then open <http://127.0.0.1:8501> locally.

### How coordination works

- Clients share results via `results/<task_name>/shared_registry.json`
  on the shared filesystem
- Before each iteration, clients merge cross-client history into their
  feedback — the LLM judge automatically sees the full model landscape
  from all clients
- The global best model is tracked across all clients
- File locking and atomic writes prevent corruption from concurrent access
- Output files include the client ID in their names (e.g. `iter0_clientexploit_run0.txt`)

## 🧪 Programmatic usage

```python
from config.schema import load_config
from gecco.prepare_data.io import load_data, split_by_participant
from gecco.prepare_data.data2text import get_data2text_function
from gecco.load_llms.model_loader import load_llm
from gecco.run_gecco import GeCCoModelSearch
from gecco.prompt_builder.prompt import PromptBuilderWrapper

cfg = load_config("config/two_step_factors_gpt54nano.yaml")

df = load_data(cfg.data.path, cfg.data.input_columns)
splits = split_by_participant(df, cfg.data.id_column, cfg.data.splits)
df_prompt, df_eval = splits["prompt"], splits["eval"]

data2text = get_data2text_function(cfg.data.data2text_function)
data_text = data2text(
    df_prompt,
    id_col=cfg.data.id_column,
    template=cfg.data.narrative_template,
    value_mappings=getattr(cfg.data, "value_mappings", None),
)

prompt_builder = PromptBuilderWrapper(cfg, data_text)

# provider="openrouter" — picks up OPENROUTER_API_KEY from .env / env
model, tokenizer = load_llm(cfg.llm.provider, cfg.llm.base_model)

search = GeCCoModelSearch(model, tokenizer, cfg, df_eval, prompt_builder)
best_model, best_bic, best_params = search.run_n_shots(run_idx=0)

print("Best Model Code:\n", best_model)
print("Best BIC:", best_bic)
print("Best Parameters:", best_params)
```

## 📊 Output

After runs, results are saved under `results/<task_name>/`:

```text
results/<task_name>/
├── models/
│   ├── best_model.py
│   ├── iter0.py
│   └── ...
├── bics/
│   ├── iter0.json
│   └── ...
├── parameters/
└── shared_registry.json   # only when running distributed
```

## 🐛 Troubleshooting

| Symptom | Likely cause |
| ------- | ------------ |
| `OPENROUTER_API_KEY not found in environment or .env file` | cwd has no `.env`, or it doesn't contain that key. `cd` into the directory with `.env`, or `source` it explicitly. |
| `401 Unauthorized` from OpenRouter | Key is present but invalid / revoked. Regenerate at <https://openrouter.ai/keys>. |
| `404 model not found` | `base_model` doesn't match an OpenRouter model id. Check the exact slug at <https://openrouter.ai/models>. |
| Hangs with no output | Some OpenRouter routes have cold-start latency on the first call (10–30s). Subsequent calls are fast. |
| SLURM clients say `OPENROUTER_API_KEY not found` but interactive runs work | The SLURM job's cwd doesn't contain `.env`. Submit from the right directory, or `source` the `.env` inside the launcher script. |

## 📄 License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📚 Citation

```bibtex
@article{rmus2025generating,
  title={Generating Computational Cognitive Models using Large Language Models},
  author={Rmus, Milena and Jagadish, Akshay K. and Mathony, Marvin and Ludwig, Tobias and Schulz, Eric},
  journal={Advances in Neural Information Processing Systems},
  year={2025},
  url={https://arxiv.org/abs/2502.00879},
}
```
