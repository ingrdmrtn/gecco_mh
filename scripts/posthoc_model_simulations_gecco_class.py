
import os, sys, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pathlib import Path
from gecco.prompt_builder.simulation_prompt import simulation_prompt
from config.schema import load_config
from gecco.prepare_data.io import load_data, split_by_participant
from gecco.load_llms.model_loader import load_llm
from gecco.run_gecco import GeCCoModelSearch
from gecco.prompt_builder.prompt import PromptBuilderWrapper


def main():
    # --- Load configuration & data ---
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(project_root / "config" / "two_step_psychiatry_individual_stai_class.yaml")
    task_name = cfg.task.name
    results_dir = project_root / "results" / "two_step_psychiatry_individual_stai_class_individual" / "models"
    data_cfg = cfg.data
    fit_type = cfg.evaluation.fit_type
    df = load_data(data_cfg.path, data_cfg.input_columns)

    for participant in df.participant.unique()[18:]:

        df_participant = df[df.participant == participant].reset_index()
        # load best mode best_model_0_participant{participant}.txtl for participant
        best_model_file = results_dir / f"best_model_0_participant{participant}.txt"
        with open(best_model_file, "r") as f:
            global_best_model = f.read()
        
        model, tokenizer = load_llm(cfg.llm.provider, cfg.llm.base_model)

        simulation_prompt_text = simulation_prompt(
            global_best_model,
            cfg,
        )
        search = GeCCoModelSearch(model, tokenizer, cfg, df_participant, None)
        simulation_text = search.generate(model, tokenizer, simulation_prompt_text)
        simulation_dir = results_dir / "simulation"
        simulation_dir.mkdir(parents=True, exist_ok=True)
        simulation_file = simulation_dir / f"simulation_model_participant{participant}.txt"
        with open(simulation_file, "w") as f:
            f.write(simulation_text)

        print(f"Simulation for participant {participant} saved to {simulation_file}")
        
if __name__ == "__main__":
    main()