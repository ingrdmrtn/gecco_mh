# examples/two_step_demo.py

import os, sys, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pathlib import Path
from gecco.prompt_builder.simulation_prompt import simulation_prompt
from config.schema import load_config
from gecco.offline_evaluation.fit_generated_models import run_fit
from gecco.prepare_data.io import load_data, split_by_participant
from gecco.prepare_data.data2text import get_data2text_function
from gecco.load_llms.model_loader import load_llm
from gecco.run_gecco import GeCCoModelSearch
from gecco.prompt_builder.prompt import PromptBuilderWrapper
import pandas as pd
import json
import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='', help='*REQUIRED* config.yaml')
    args = parser.parse_args()
    # --- Load configuration & data ---
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(project_root / "config" / args.config)
    data_cfg = cfg.data
    metadata = getattr(getattr(cfg, "metadata", None), "flag", False)
    max_independent_runs  = cfg.loop.max_independent_runs

    df = load_data(data_cfg.path, data_cfg.input_columns)
    splits = split_by_participant(df, data_cfg.id_column, data_cfg.splits)
    df_prompt = splits["prompt"]

    # --- Split eval/test by proportion of remaining participants ---
    eval_test_proportion = getattr(cfg.evaluation, "eval_test_split", 0.7)
    non_prompt_ids = sorted(set(df[data_cfg.id_column].unique()) - set(df_prompt[data_cfg.id_column].unique()))
    np.random.seed(getattr(cfg.evaluation, "split_seed", 42))
    np.random.shuffle(non_prompt_ids)
    split_idx = int(len(non_prompt_ids) * eval_test_proportion)
    eval_ids = non_prompt_ids[:split_idx]
    test_ids = non_prompt_ids[split_idx:]
    df_eval = df[df[data_cfg.id_column].isin(eval_ids)]
    df_test = df[df[data_cfg.id_column].isin(test_ids)]
    print(f"[GeCCo] Eval/test split: {len(eval_ids)} eval, {len(test_ids)} test "
          f"({eval_test_proportion:.0%}/{1-eval_test_proportion:.0%})")

    if getattr(cfg.loop, "early_stopping", "False") == "True":
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
        metadata=getattr(cfg.metadata, "narrative_template", None) if metadata else None,
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
        best_iter = search.best_iter
        # --- Print results for a run ---
        print(f"\n[🏁 GeCCo] Run {r} complete.")
        print(f"Best model parameters: {', '.join(best_params)}")
        print(f"Best mean BIC: {best_bic:.2f}")
        print(f"Best params: {', '.join(best_params)}")

        if cfg.llm.do_simulation == "True":
            from gecco.prompt_builder.simulation_prompt import simulation_prompt

            simulation_prompt_text = simulation_prompt(
                best_model,
                cfg,
            )
            simulation_text = search.generate(model, tokenizer, simulation_prompt_text)
            simulation_dir = search.results_dir / "simulation"
            simulation_dir.mkdir(parents=True, exist_ok=True)
            simulation_file = simulation_dir / f"simulation_model_run{r}.txt"
            with open(simulation_file, "w") as f:
                f.write(simulation_text)
    
        # fit the best model to test data: (1) report BIC, (2) save best params, (3) save simulation
        print("\n Fitting best model to test data...")
        try:
            # extract function name from model code
            func_name = f"cognitive_model{best_iter}" 
            fit_res = run_fit(df_test, best_model, cfg=cfg, expected_func_name=func_name)
            mean_metric = float(fit_res["metric_value"])
            metric_name = fit_res["metric_name"]
            params = fit_res["param_names"]

            print(f"[GeCCo] {func_name}: mean {metric_name} = {mean_metric:.2f}")
            
            # save best model bic
            results_dir = search.results_dir
            results_dir.mkdir(parents=True, exist_ok=True)
            best_bic_file = (
                results_dir / "bics" / f"best_bic_on_test_run{r}.json"
            )
            # save best bics to json across participants
            # fit_res['eval_metrics'] is a list of bics for each participant: [437.4235310062762, 557.8133717150291, 478.1469470482797, 470.78509749717364, 328.51242771038415, 219.52379669611818, 377.2421201048371, 351.9631188780421, 337.3545037733362, 366.12602047624944, 450.09353867481525, 126.98829795881838, 253.5581869957461, 406.95593007888044, 380.5687939943248, 264.2553031674379, 514.3963045824119, 384.2459136412227, 336.4981903599349, 396.85986980687835, 372.5575441074137, 329.24768724723754, 457.22504147884547, 456.06518231837924, 382.5115132329936, 325.06967212781683, 446.54528039644237, 278.15614188014604, 334.1230336957856, 532.6872059457748, 434.50828260351295]
            with open(best_bic_file, "w") as f:
                json.dump({"mean_"+metric_name: mean_metric, "individual_"+metric_name: fit_res['eval_metrics']}, f)    

            # save best model params
            param_df = pd.DataFrame(
                fit_res["parameter_values"],
                columns=params
            )
            param_dir = results_dir / "parameters"
            param_dir.mkdir(parents=True, exist_ok=True)
            param_file = (
                param_dir / f"best_params_on_test_run{r}.csv"
            )
            param_df.to_csv(param_file, index=False)

        except Exception as e:
            print(f"[⚠️ GeCCo] Error fitting {func_name}: {e}")

        if best_bic < global_best_bic:
            global_best_bic = best_bic
            global_best_model = best_model
            global_best_params = best_params
            global_best_iter = best_iter


    # --- Print final results ---
    print("\n[🏁 GeCCo] Search complete.")
    print(f"Best mean BIC: {global_best_bic:.2f}")

# -------------------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()
