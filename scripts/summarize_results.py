"""Derive report-ready statistics and chart data from the raw experiment files."""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def main():
    evaluation = load("evaluation.json")
    pooled = {}
    for scope in ("held_out", "out_of_distribution"):
        pooled[scope] = {}
        for method in dict.fromkeys(r["method"] for r in evaluation["rollouts"]):
            rows = [r for r in evaluation["rollouts"] if r["scope"] == scope and r["method"] == method]
            pooled[scope][method] = {"trajectories": len(rows),
                "theta_rmse": math.sqrt(sum(r["theta_rmse"] ** 2 for r in rows) / len(rows)),
                "omega_rmse": math.sqrt(sum(r["omega_rmse"] ** 2 for r in rows) / len(rows)),
                "max_abs_theta_error": max(r["max_abs_theta_error"] for r in rows)}
    trajectories = {}
    with (RESULTS / "rollouts.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            trajectory = int(row["trajectory"])
            if trajectory not in (68, 74, 80):
                continue
            if round(float(row["t"]) / 0.02) % 5:
                continue
            curves = trajectories.setdefault(str(trajectory), {})
            curves.setdefault(row["method"], []).append({k: float(row[k]) for k in
                ("t", "theta", "omega", "reference_theta", "reference_omega", "theta_error", "omega_error", "energy")})
    histories = {}
    for name in ("main", "tiny"):
        with (RESULTS / f"history_{name}.csv").open(newline="", encoding="utf-8") as stream:
            histories[name] = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(stream)]
    models = {name: json.loads((ROOT / "models" / f"{name}.json").read_text(encoding="utf-8")) for name in ("main", "tiny")}
    output = {"one_step": evaluation["one_step"], "rollouts_pooled": pooled,
        "improvement_fraction": evaluation["standardized_improvement_fraction"],
        "acceptance_passed": evaluation["acceptance_passed"], "dataset": load("dataset.json"),
        "training": {name: model["training"] for name, model in models.items()},
        "normalization": {"input": models["main"]["input_normalization"], "delta": models["main"]["delta_normalization"]},
        "benchmark": {"rust": load("benchmark_rust.json"), "numpy": load("benchmark_numpy.json")},
        "parity": load("parity.json"), "trajectories": trajectories, "histories": histories}
    (RESULTS / "summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("Generated report summary from raw CSV/JSON evidence")


if __name__ == "__main__":
    main()
