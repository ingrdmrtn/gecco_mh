# examples/two_step_demo.py

import os, sys, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pathlib import Path

from config.schema import load_config
from gecco.prepare_data.io import load_data, split_by_participant
from gecco.prepare_data.data2text import get_data2text_function
from gecco.load_llms.model_loader import load_llm
from gecco.run_gecco import GeCCoModelSearch
from gecco.prompt_builder.prompt import PromptBuilderWrapper

def main():
    # --- Load configuration & data ---
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(project_root / "config" / "two_step_psychiatry_group_metadata.yaml")
    data_cfg = cfg.data
    metadata = cfg.metadata.flag
    max_independent_runs  = cfg.loop.max_independent_runs

    df = load_data(data_cfg.path, data_cfg.input_columns)
    splits = split_by_participant(df, data_cfg.id_column, data_cfg.splits)
    df_prompt, df_eval = splits["prompt"], splits["eval"]

    if cfg.loop.early_stopping == "True":
        df_baselines = load_data(data_cfg.path)
        splits_baselines = split_by_participant(df_baselines, data_cfg.id_column, data_cfg.splits)
        df_prompt_splits, df_eval_splits = splits_baselines["prompt"], splits_baselines["eval"]
        baseline_bic = np.mean(df_eval_splits.baseline_bic)

    else:
        baseline_bic = None

    # --- Convert data to narrative text for the LLM ---
    data2text = get_data2text_function(data_cfg.data2text_function)
    data_text = data2text(
        df_prompt,
        id_col=data_cfg.id_column,
        template=data_cfg.narrative_template,
        fit_type=getattr(cfg.evaluation, "fit_type", "group"),
        metadata =cfg.metadata.narrative_template if metadata else None,
        max_trials=getattr(data_cfg, "max_prompt_trials", None),
        value_mappings=getattr(data_cfg, "value_mappings", None)  # 👈 add this
    )

    # --- Build prompt builder ---
    prompt_builder = PromptBuilderWrapper(cfg, data_text, df_prompt)

    # --- Load LLM ---
    model, tokenizer = load_llm(cfg.llm.provider, cfg.llm.base_model)

    # --- Run GeCCo iterative model search ---
    search = GeCCoModelSearch(model, tokenizer, cfg, df_eval, prompt_builder)
    global_best_model = None
    global_best_bic = np.inf
    global_best_params = None

    for r in range(max_independent_runs):
        best_model, best_bic, best_params = search.run_n_shots(r, baseline_bic=baseline_bic)

        if best_bic < global_best_bic:
            global_best_bic = best_bic
            global_best_model = best_model
            global_best_params = best_params

            # --- Print final results ---
            print("\n[🏁 GeCCo] Search complete.")
            print(f"Best model parameters: {', '.join(best_params)}")
            print(f"Best mean BIC: {best_bic:.2f}")
            print(f"Best params: {', '.join(global_best_params)}")


    # --- Print final results ---
    print("\n[🏁 GeCCo] Search complete.")
    print(f"Best model parameters: {', '.join(best_params)}")
    print(f"Best mean BIC: {best_bic:.2f}")


# -------------------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()
