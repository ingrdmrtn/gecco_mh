"""CLI route for distributed launch."""

from __future__ import annotations

import argparse
from pathlib import Path
from collections.abc import Sequence

from config.schema import load_config
from gecco.cli.launcher_utils import LaunchCommand, LaunchExecutor, LaunchPlan, SubmissionResult
from gecco.load_llms.provider_registry import get_provider_spec
from gecco.sentry_init import init_sentry


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _print_submission_result(result: SubmissionResult) -> None:
    label_map = {
        "client_array": "Client array",
        "generator": "Generator",
        "evaluator": "Evaluator",
        "orchestrator": "Orchestrator",
        "test_evaluation": "Test evaluation",
    }
    display_label = label_map.get(result.label, result.label)
    if result.job_id:
        print(f"  {display_label} job ID: {result.job_id}")
    print()


def _join_command(parts: Sequence[str]) -> str:
    return " ".join(part for part in parts if part)


def _optional_vllm_flag(vllm_url: str | None) -> str:
    return f'--vllm-url "{vllm_url}"' if vllm_url else ""


def _optional_conda_arg(conda_env: str | None) -> str:
    return f'"{conda_env}"' if conda_env else ""


def _print_local_command(label: str, command: str) -> None:
    print(f"[{label}] {command}")


def _get_cmg_state(cfg):
    cmg_cfg = getattr(cfg, "centralized_model_generation", None)
    return cmg_cfg is not None and getattr(cmg_cfg, "enabled", False), cmg_cfg


def _build_regular_launch_plan(
    *,
    config: str,
    profiles_csv: str,
    array_spec: str,
    results_dir_rel: str,
    resolved_cpus_per_task: int,
    partition_flag: str,
    mem_flag: str,
    vllm_url: str | None,
    conda_env: str | None,
    resolved_launch_orchestrator: bool,
    n_clients: int | None,
) -> LaunchPlan:
    vllm_arg = _optional_vllm_flag(vllm_url)
    conda_arg = _optional_conda_arg(conda_env)
    commands: list[LaunchCommand] = [
        LaunchCommand(
            label="client_array",
            command=_join_command(
                [
                    "sbatch",
                    f"--array={array_spec}",
                    f"--cpus-per-task={resolved_cpus_per_task}",
                    partition_flag,
                    mem_flag,
                    "bash/run_gecco_distributed.sh",
                    f'"{config}"',
                    f'"{profiles_csv}"',
                    vllm_arg,
                    conda_arg,
                ]
            ),
        )
    ]

    if resolved_launch_orchestrator:
        n_clients_arg = f'"{n_clients}"' if n_clients is not None else ""
        commands.append(
            LaunchCommand(
                label="orchestrator",
                command=_join_command(
                    [
                        "sbatch",
                        "--cpus-per-task=8",
                        partition_flag,
                        "--mem=16G",
                        "bash/run_judge_orchestrator.sh",
                        f'"{config}"',
                        vllm_arg,
                        n_clients_arg,
                        conda_arg,
                    ]
                ),
            )
        )

    commands.append(
        LaunchCommand(
            label="test_evaluation",
            command=_join_command(
                [
                    "sbatch",
                    "{dependency}",
                    "--cpus-per-task=8",
                    partition_flag,
                    "--mem=16G",
                    "bash/run_test_evaluation.sh",
                    f'"{config}"',
                    f'"{results_dir_rel}"',
                    conda_arg,
                ]
            ),
            dependency_labels=("client_array",),
            required_dependency_labels=("client_array",),
        )
    )

    return LaunchPlan(commands=tuple(commands))


def _build_cmg_launch_plan(
    *,
    config: str,
    generator_client: str,
    n_models: int,
    results_dir_rel: Path,
    resolved_cpus_per_task: int,
    partition_flag: str,
    mem_flag: str,
    resolved_vllm_url: str,
    conda_env: str | None,
    final_eval_enabled: bool,
) -> LaunchPlan:
    conda_arg = _optional_conda_arg(conda_env)
    commands: list[LaunchCommand] = [
        LaunchCommand(
            label="generator",
            command=_join_command(
                [
                    "sbatch",
                    "--job-name=gecco-cmg-generator",
                    f"--cpus-per-task={resolved_cpus_per_task}",
                    partition_flag,
                    mem_flag,
                    "--output=logs/gecco-cmg-generator-%j.out",
                    "--error=logs/gecco-cmg-generator-%j.err",
                    str(PROJECT_ROOT / "bash/run_cmg_generator.sh"),
                    f'"{config}"',
                    f'"{generator_client}"',
                    f'"{resolved_vllm_url}"',
                    conda_arg,
                ]
            ),
        ),
        LaunchCommand(
            label="evaluator",
            command=_join_command(
                [
                    "sbatch",
                    f"--array=0-{n_models - 1}",
                    "--job-name=gecco-cmg-evaluator",
                    f"--cpus-per-task={resolved_cpus_per_task}",
                    partition_flag,
                    mem_flag,
                    "--output=logs/gecco-cmg-evaluator-%A_%a.out",
                    "--error=logs/gecco-cmg-evaluator-%A_%a.err",
                    str(PROJECT_ROOT / "bash/run_cmg_evaluator.sh"),
                    f'"{config}"',
                    f'"{resolved_vllm_url}"',
                    conda_arg,
                ]
            ),
        ),
        LaunchCommand(
            label="orchestrator",
            command=_join_command(
                [
                    "sbatch",
                    "--job-name=gecco-cmg-orchestrator",
                    "--cpus-per-task=8",
                    partition_flag,
                    "--mem=16G",
                    "--output=logs/gecco-cmg-orchestrator-%j.out",
                    "--error=logs/gecco-cmg-orchestrator-%j.err",
                    str(PROJECT_ROOT / "bash/run_judge_orchestrator.sh"),
                    f'"{config}"',
                    f'"{resolved_vllm_url}"',
                    f'"{n_models}"',
                    conda_arg,
                ]
            ),
        ),
    ]

    if final_eval_enabled:
        commands.append(
            LaunchCommand(
                label="final_eval",
                command=_join_command(
                    [
                        "sbatch",
                    "{dependency}",
                    "--cpus-per-task=8",
                    partition_flag,
                    "--mem=16G",
                    str(PROJECT_ROOT / "bash/run_test_evaluation.sh"),
                    f'"{config}"',
                    f'"{results_dir_rel}"',
                    conda_arg,
                ]
            ),
                dependency_labels=("generator", "evaluator", "orchestrator"),
                dependency_fallback="--dependency=afterok:<generator_job_id>:<evaluator_job_id>:<orchestrator_job_id>",
            )
        )

    return LaunchPlan(commands=tuple(commands))


def _print_regular_local_preview(
    *,
    config: str,
    profiles: list[str],
    extra_clients: int,
    vllm_url: str | None,
    resolved_launch_orchestrator: bool,
    n_clients: int | None,
) -> None:
    vllm_arg = _optional_vllm_flag(vllm_url)
    all_profiles = profiles + [""] * extra_clients
    print("[Local preview] Distributed client commands")
    for client_id, profile in enumerate(all_profiles):
        if profile:
            command = _join_command(
                [
                    "python -m gecco internal distributed-client",
                    f'--config "{config}"',
                    f'--client-profile "{profile}"',
                    vllm_arg,
                ]
            )
        else:
            command = _join_command(
                [
                    "python -m gecco internal distributed-client",
                    f'--config "{config}"',
                    f"--client-id {client_id}",
                    vllm_arg,
                ]
            )
        _print_local_command(f"Client {client_id}", command)

    if resolved_launch_orchestrator:
        n_clients_arg = f"--n-clients {n_clients}" if n_clients is not None else ""
        command = _join_command(
            [
                "python -m gecco internal judge-orchestrate",
                f'--config "{config}"',
                vllm_arg,
                n_clients_arg,
            ]
        )
        _print_local_command("Orchestrator", command)


def _print_cmg_local_preview(
    *,
    config: str,
    generator_client: str,
    n_models: int,
    results_dir_rel: Path,
    resolved_vllm_url: str,
    final_eval_enabled: bool,
) -> None:
    vllm_arg = _optional_vllm_flag(resolved_vllm_url or None)
    print("[Local preview] CMG distributed commands")
    _print_local_command(
        "Generator",
        _join_command(
            [
                "python -m gecco internal distributed-client",
                f'--config "{config}"',
                f'--client-profile "{generator_client}"',
                vllm_arg,
            ]
        ),
    )
    for index in range(n_models):
        _print_local_command(
            f"Evaluator {index}",
            _join_command(
                [
                    "python -m gecco internal distributed-client",
                    f'--config "{config}"',
                    f"--client-id {index}",
                    vllm_arg,
                ]
            ),
        )
    _print_local_command(
        "Orchestrator",
        _join_command(
            [
                "python -m gecco internal judge-orchestrate",
                f'--config "{config}"',
                vllm_arg,
                f'--n-clients {n_models}',
            ]
        ),
    )
    if final_eval_enabled:
        _print_local_command(
            "Final evaluation",
            _join_command(
                [
                    "python -m gecco internal test-evaluation",
                    f'--config "{config}"',
                    f'--results-dir "{results_dir_rel}"',
                    "--write-store",
                ]
            ),
        )


def register_parser(subparsers) -> argparse.ArgumentParser:
    """Register the distributed launcher subcommand."""
    parser = subparsers.add_parser("distributed", help="Launch a distributed GeCCo run")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--profiles", type=str, default=None)
    parser.add_argument("--extra-clients", type=int, default=0)
    parser.add_argument("--vllm-url", type=str, default=None)
    parser.add_argument("--conda-env", type=str, default=None)
    parser.add_argument("--partition", type=str, default=None)
    parser.add_argument("--cpus-per-task", type=int, default=None)
    parser.add_argument("--mem", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--launch-orchestrator", action="store_true")
    parser.set_defaults(handler=main)
    return parser


def get_profiles_from_config(config_path):
    """Read client profile names from a YAML config file."""
    cfg = load_config(config_path)
    clients = getattr(cfg, "clients", {}) or {}
    if not clients:
        return []
    return list(clients.keys())


def run_distributed_launcher(
    *,
    config: str,
    profiles: str | None = None,
    extra_clients: int = 0,
    vllm_url: str | None = None,
    conda_env: str | None = None,
    partition: str | None = None,
    cpus_per_task: int | None = None,
    mem: str | None = None,
    dry_run: bool = False,
    local: bool = False,
    launch_orchestrator: bool = False,
) -> int | None:
    """Launch a distributed GeCCo search from a config file."""
    config_path = PROJECT_ROOT / "config" / config

    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        raise SystemExit(1)

    cfg = load_config(config_path)

    init_sentry(
        task_name=cfg.task.name,
        config_name=config,
    )

    provider = cfg.llm.provider
    try:
        provider_spec = get_provider_spec(provider)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc
    slurm_cfg = getattr(cfg, "slurm", {}) or {}
    resolved_cpus_per_task = cpus_per_task or slurm_cfg.get("cpus_per_task", 48)
    resolved_mem = mem or slurm_cfg.get("mem_per_task")
    mem_flag = f"--mem={resolved_mem}" if resolved_mem else ""

    resolved_partition = partition or slurm_cfg.get("partition")
    partition_flag = f"--partition={resolved_partition}" if resolved_partition else ""

    cmg_enabled, cmg_cfg = _get_cmg_state(cfg)
    resolved_launch_orchestrator = launch_orchestrator or getattr(cfg, "judge", None) is not None
    conda_arg = _optional_conda_arg(conda_env)

    if cmg_enabled:
        generator_client = str(getattr(cmg_cfg, "generator_client", ""))
        if not generator_client:
            print("ERROR: centralized_model_generation.generator_client is required")
            raise SystemExit(1)
        if generator_client.isdigit() or generator_client.lstrip("-").isdigit():
            print(
                "ERROR: centralized_model_generation.generator_client must be a named profile, not a numeric evaluator ID"
            )
            raise SystemExit(1)

        n_models = getattr(cmg_cfg, "n_models", None)
        if not isinstance(n_models, int) or n_models <= 0:
            print("ERROR: centralized_model_generation.n_models must be a positive integer")
            raise SystemExit(1)

        final_eval_enabled = getattr(cmg_cfg, "run_final_evaluation", True)
        resolved_vllm_url = vllm_url or ""
        task_name = getattr(cfg.task, "name", "unknown")
        results_dir = PROJECT_ROOT / "results" / task_name
        if getattr(cfg.evaluation, "fit_type", "group") == "individual":
            results_dir = PROJECT_ROOT / "results" / f"{task_name}_individual"
        results_dir_rel = results_dir.relative_to(PROJECT_ROOT)

        print(f"Config:            {config}")
        print(f"Provider:          {provider_spec.label} ({provider_spec.key})")
        print(f"CPUs/task:         {resolved_cpus_per_task}")
        if resolved_mem:
            print(f"Memory:            {resolved_mem}")
        print(f"Generator client:  {generator_client}")
        print(f"Evaluators:        {n_models}")
        print(f"Final eval:        {'enabled' if final_eval_enabled else 'disabled'}")
        print(f"vLLM URL:          {resolved_vllm_url or '(from env / .vllm_env)'}")
        print()

        if local:
            _print_cmg_local_preview(
                config=config,
                generator_client=generator_client,
                n_models=n_models,
                results_dir_rel=results_dir_rel,
                resolved_vllm_url=resolved_vllm_url,
                final_eval_enabled=final_eval_enabled,
            )
            return None

        executor = LaunchExecutor()
        plan = _build_cmg_launch_plan(
            config=config,
            generator_client=generator_client,
            n_models=n_models,
            results_dir_rel=results_dir_rel,
            resolved_cpus_per_task=resolved_cpus_per_task,
            partition_flag=partition_flag,
            mem_flag=mem_flag,
            resolved_vllm_url=resolved_vllm_url,
            conda_env=conda_env,
            final_eval_enabled=final_eval_enabled,
        )
        submission_results = executor.execute(plan, dry_run=dry_run, on_result=_print_submission_result)
        results_by_label = {result.label: result for result in submission_results}
        if final_eval_enabled and results_by_label.get("final_eval") and results_by_label["final_eval"].job_id:
            print(
                f"Final evaluation will run after all CMG jobs complete: job {results_by_label['final_eval'].job_id}"
            )
        print("Launched successfully. Monitor with:")
        print(f"  python -m gecco monitor --task {task_name} --watch 10")
        return None

    resolved_profiles = profiles.split(",") if profiles else list((cfg.clients or {}).keys())
    n_profiled = len(resolved_profiles)
    n_total = n_profiled + extra_clients
    if n_total == 0:
        print(
            "ERROR: No profiles found in config and --extra-clients is 0. Nothing to launch."
        )
        raise SystemExit(1)

    all_profiles = resolved_profiles + [""] * extra_clients
    profiles_csv = ",".join(all_profiles)
    array_spec = f"0-{n_total - 1}"
    n_clients = getattr(cfg.loop, "n_clients", None)
    task_name = cfg.task.name
    results_dir = PROJECT_ROOT / "results" / task_name
    if getattr(cfg.evaluation, "fit_type", "group") == "individual":
        results_dir = PROJECT_ROOT / "results" / f"{task_name}_individual"
    results_dir_rel = str(results_dir.relative_to(PROJECT_ROOT))

    print(f"Config:            {config}")
    print(f"Provider:          {provider_spec.label} ({provider_spec.key})")
    print(f"CPUs/task:         {resolved_cpus_per_task}")
    if resolved_mem:
        print(f"Memory:            {resolved_mem}")
    print(f"Profiles:          {resolved_profiles if resolved_profiles else '(none)'}")
    print(f"Extra clients:     {extra_clients}")
    print(f"Total clients:     {n_total}")
    print(f"Array spec:        --array={array_spec}")
    print(f"Profiles CSV:      {profiles_csv}")
    if resolved_partition:
        print(f"Partition:         {resolved_partition}")
    if provider_spec.key == "vllm":
        print(f"vLLM URL:          {vllm_url or '(from env / .vllm_env)'}")
    if resolved_launch_orchestrator:
        print("[Orchestrator]     ENABLED (centralized judge)")
        if n_clients:
            print(f"[Orchestrator]     n_clients: {n_clients}")
    print()

    if local:
        _print_regular_local_preview(
            config=config,
            profiles=resolved_profiles,
            extra_clients=extra_clients,
            vllm_url=vllm_url,
            resolved_launch_orchestrator=resolved_launch_orchestrator,
            n_clients=n_clients,
        )
        return None

    executor = LaunchExecutor()
    plan = _build_regular_launch_plan(
        config=config,
        profiles_csv=profiles_csv,
        array_spec=array_spec,
        results_dir_rel=results_dir_rel,
        resolved_cpus_per_task=resolved_cpus_per_task,
        partition_flag=partition_flag,
        mem_flag=mem_flag,
        vllm_url=vllm_url,
        conda_env=conda_env,
        resolved_launch_orchestrator=resolved_launch_orchestrator,
        n_clients=n_clients,
    )
    main_results = executor.execute(plan, dry_run=dry_run, on_result=_print_submission_result)
    results_by_label = {result.label: result for result in main_results}

    print("Launched successfully. Monitor with:")
    task_name = cfg.task.name
    print(f"  python -m gecco monitor --task {task_name} --watch 10")
    test_eval_result = results_by_label.get("test_evaluation")
    if test_eval_result and test_eval_result.job_id:
        test_eval_job_id = test_eval_result.job_id
        print(
            f"Test evaluation will run after all clients complete: job {test_eval_job_id}"
        )
    return None


def main(args: argparse.Namespace) -> int | None:
    """Run the distributed launcher from parsed CLI arguments."""
    return run_distributed_launcher(
        config=args.config,
        profiles=args.profiles,
        extra_clients=args.extra_clients,
        vllm_url=args.vllm_url,
        conda_env=args.conda_env,
        partition=args.partition,
        cpus_per_task=args.cpus_per_task,
        mem=args.mem,
        dry_run=args.dry_run,
        local=args.local,
        launch_orchestrator=args.launch_orchestrator,
    )
