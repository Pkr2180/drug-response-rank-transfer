import argparse
from pathlib import Path

from research_common.io import managed_run, read_config, read_table, write_json, write_table
from research_common.plots import audit_scatter

from .benchmark import run_benchmark, validate_config
from .data import synthetic_compounds, validate
from .prioritization import pareto_candidates


def main():
    parser = argparse.ArgumentParser(description="External-validity meta-evaluation; no clinical claims")
    subs = parser.add_subparsers(dest="command", required=True)
    for command in ["demo", "run"]:
        p = subs.add_parser(command)
        p.add_argument("--config", default=str(Path(__file__).parent / "configs" / "demo.json"))
        p.add_argument("--output", required=True)
        if command == "run":
            p.add_argument("--input", required=True)
            p.add_argument("--provenance", required=True, help="Completed data/feature provenance JSON")
    p = subs.add_parser("validate")
    p.add_argument("--input", required=True)
    p = subs.add_parser("prioritize")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "validate":
        validate(read_table(args.input))
        print("Input contract passed; provenance and biological validity still require review")
        return
    if args.command == "prioritize":
        if Path(args.output).exists():
            raise FileExistsError(args.output)
        pareto_candidates(read_table(args.input)).to_csv(args.output, index=False, mode="x")
        return
    config = read_config(args.config)
    validate_config(config)
    inputs = [args.config]
    demo = args.command == "demo"
    if demo:
        frame = synthetic_compounds(config.get("data_seed", 17))
        provenance = {"data_origin": "SYNTHETIC DEMONSTRATION"}
    else:
        from research_common.provenance import validate_provenance

        provenance = read_config(args.provenance)
        validate_provenance(provenance)
        frame = read_table(args.input)
        inputs.extend([args.input, args.provenance])
    validate(frame, allow_synthetic=demo)
    status = "SYNTHETIC_DEMONSTRATION" if demo else "USER_SUPPLIED_DATA_UNVERIFIED"
    with managed_run(args.output, config, inputs, status, provenance) as output:
        results = run_benchmark(
            frame, config, allow_synthetic=demo, progress=lambda text: print(text, flush=True)
        )
        for name, table in results.items():
            write_table(output / (name + ".csv"), table)
        if demo:
            write_table(output / "synthetic_input.csv", frame)
        audit_scatter(
            results["scores"],
            "internal_compound_brier",
            "external_compound_brier",
            output / "external_validity.png",
            status + "\nInternal versus external Brier score",
        )
        write_json(
            output / "interpretation.json",
            {
                "status": status,
                "primary_metric": "equal-compound mean Brier loss",
                "interval": "95% percentile bootstrap at protocol grouping level, conditional on the fitted model; few blocks may yield unstable intervals",
                "selection_ties": "Uniform selection among exact internal-score ties",
                "fusion_objective": "equal-compound calibration Brier loss",
                "meta_metrics": ["absolute performance discrepancy", "rank agreement", "selection regret"],
                "important": "Repeated splits share observations: do not treat rows as independent experiments.",
                "scope": "Toxicity endpoint evaluation does not validate periodontitis efficacy or clinical safety.",
            },
        )
    print(f"Completed {status}: {output.resolve()}")


if __name__ == "__main__":
    main()
