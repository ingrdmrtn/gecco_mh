# examples/two_step_demo.py

import os, sys, numpy as np

from gecco.prompt_builder.simulation_prompt import simulation_prompt

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
    cfg = load_config(project_root / "config" / "rlwm_individual.yaml")
    data_cfg = cfg.data
    fit_type = cfg.evaluation.fit_type
    max_independent_runs  = cfg.loop.max_independent_runs

    df = load_data(data_cfg.path, data_cfg.input_columns)
    df = df[df.blocks < 5]

    # sample several younger and several older participants
    df_w_participant = load_data(data_cfg.path)
    young_participants = list(df_w_participant[df_w_participant.age<45].participant.unique()[:15])
    old_participants = list(df_w_participant[df_w_participant.age>45].participant.unique()[:15])
    all_participants  = young_participants + old_participants
    all_participants = all_participants [17:]
    if fit_type == "individual":


        # --- Convert data to narrative text for the LLM ---
        data2text = get_data2text_function(data_cfg.data2text_function)

        for participant in all_participants:

            # best_hybrid_bic =
            df_participant = df[df.participant == participant].reset_index()
            df_participant = df_participant[df_participant.rewards >= 0].reset_index()
            if cfg.loop.early_stopping == "True":
                baseline_bics = load_data(data_cfg.path)
                baseline_bic = baseline_bics[baseline_bics.participant == participant].reset_index().baseline_bic[0]
            else:
                baseline_bic = None

            data_text = data2text(
                df_participant,
                id_col=data_cfg.id_column,
                template=data_cfg.narrative_template,
                fit_type = fit_type,
                max_trials=getattr(data_cfg, "max_prompt_trials", None),
                max_blocks=getattr(data_cfg, "max_prompt_blocks", None),
                value_mappings=getattr(data_cfg, "value_mappings", None)
            )

            # --- Build prompt builder ---
            prompt_builder = PromptBuilderWrapper(cfg, data_text, df_participant)

            # --- Load LLM ---
            model, tokenizer = load_llm(cfg.llm.provider, cfg.llm.base_model)

            # --- Run GeCCo iterative model search ---
            search = GeCCoModelSearch(model, tokenizer, cfg, df_participant, prompt_builder)
            global_best_model = None
            global_best_bic = np.inf
            global_best_params = None

            for r in range(max_independent_runs):
                best_model, best_bic, best_params = search.run_n_shots(r, baseline_bic)

                if best_bic < global_best_bic:
                    global_best_bic = best_bic
                    global_best_model = best_model
                    global_best_params = best_params

                # --- Print final results ---
                print("\n[🏁 GeCCo] Search complete.")
                print(f"Best model parameters: {', '.join(best_params)}")
                print(f"Best mean BIC: {best_bic:.2f}")

            # --- Print final results ---
            print("\n[🏁 GeCCo] Final best across runs")
            print(f"Participant {participant}")
            print(f"Best BIC: {global_best_bic:.2f}")
            print(f"Best params: {', '.join(global_best_params)}")

            if cfg.llm.do_simulation == "True":
                from gecco.prompt_builder.simulation_prompt import simulation_prompt

                simulation_prompt_text = simulation_prompt(
                    global_best_model,
                    cfg
                )
                simulation_text = search.generate(model, tokenizer, simulation_prompt_text)
                simulation_dir = search.results_dir / "simulation"
                simulation_dir.mkdir(parents=True, exist_ok=True)
                simulation_file = simulation_dir / f"simulation_model_participant{participant}.txt"
                with open(simulation_file, "w") as f:
                    f.write(simulation_text)





    else:
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
            fit_type=fit_type,
            max_trials=getattr(data_cfg, "max_prompt_trials", None),
            max_blocks= getattr(data_cfg, "max_prompt_blocks", None),
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
            best_model, best_bic, best_params = search.run_n_shots(r,baseline_bic)

            if best_bic < global_best_bic:
                global_best_bic = best_bic
                global_best_model = best_model
                global_best_params = best_params

            # --- Print final results ---
            print("\n[🏁 GeCCo] Search complete.")
            print(f"Best model parameters: {', '.join(best_params)}")
            print(f"Best mean BIC: {best_bic:.2f}")
            print(f"Best params: {', '.join(global_best_params)}")

        if cfg.llm.do_simulation == "True":
            from gecco.prompt_builder.simulation_prompt import simulation_prompt

            simulation_prompt_text = simulation_prompt(
                global_best_model,
                cfg,
            )
            simulation_text = search.generate(model, tokenizer, simulation_prompt_text)
            simulation_dir = search.results_dir / "simulation"
            simulation_dir.mkdir(parents=True, exist_ok=True)
            simulation_file = simulation_dir / f"simulation_model.txt"
            with open(simulation_file, "w") as f:
                f.write(simulation_text)

        # --- Print final results ---
        print("\n[🏁 GeCCo] Search complete.")
        print(f"Best model parameters: {', '.join(best_params)}")
        print(f"Best mean BIC: {best_bic:.2f}")

# -------------------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()
