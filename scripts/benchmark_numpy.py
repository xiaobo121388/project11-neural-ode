"""Manual NumPy implementation of the identical f64 SGD workload and parity test.

Run AFTER: project11 benchmark --out ROOT
No PyTorch, TensorFlow, autodiff, validation passes, I/O or startup inside timing.
"""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import copy
import csv
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np


def array(item):
    return np.asarray(item["data"], dtype=np.float64).reshape(item["dim"])


def network(item):
    return [{"weights": array(l["weights"]), "bias": array(l["bias"]), "activation": l["activation"]}
            for l in item["layers"]]


def activate(z, name):
    if name == "Tanh":
        return np.tanh(z)
    if name == "Relu":
        return np.maximum(z, 0.0)
    if name == "Sigmoid":
        out = np.empty_like(z)
        positive = z >= 0
        out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
        e = np.exp(z[~positive])
        out[~positive] = e / (1.0 + e)
        return out
    if name == "Linear":
        return z
    raise ValueError("Unknown activation")


def forward(net, x):
    activations = [x]
    preactivations = []
    for layer in net:
        z = activations[-1] @ layer["weights"] + layer["bias"]
        preactivations.append(z)
        activations.append(activate(z, layer["activation"]))
    return activations, preactivations


def gradients(net, x, y):
    activations, preactivations = forward(net, x)
    residual = activations[-1] - y
    loss = np.mean(residual * residual)
    delta = residual * (2.0 / residual.size)
    grads = []
    for i in range(len(net) - 1, -1, -1):
        name = net[i]["activation"]
        z = preactivations[i]
        if name == "Tanh":
            delta = delta * (1.0 - np.tanh(z) ** 2)
        elif name == "Sigmoid":
            a = activate(z, name)
            delta = delta * a * (1.0 - a)
        elif name == "Relu":
            delta = delta * (z > 0.0)
        grads.append((activations[i].T @ delta, delta.sum(axis=0)))
        if i > 0:
            delta = delta @ net[i]["weights"].T
    grads.reverse()
    return float(loss), grads


def update(net, x, y, rate):
    loss, grads = gradients(net, x, y)
    for layer, (dw, db) in zip(net, grads):
        layer["weights"] += -rate * dw
        layer["bias"] += -rate * db
    return loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    fixture_dir = args.root / "benchmark_fixture"
    with (fixture_dir / "shared.json").open(encoding="utf-8") as stream:
        fixture = json.load(stream)
    if fixture["schema_version"] != 1:
        raise ValueError("Unsupported fixture schema")
    x, y = array(fixture["x"]), array(fixture["y"])
    initial = network(fixture["initial"])
    orders = [np.asarray(v, dtype=np.intp) for v in fixture["orders"]]
    rate, batch = fixture["learning_rate"], fixture["batch_size"]
    expected = json.loads((fixture_dir / "parity_rust.json").read_text(encoding="utf-8"))
    loss, grads = gradients(initial, x[:8], y[:8])
    predicted = forward(initial, x[:8])[0][-1]
    updated = copy.deepcopy(initial)
    update(updated, x[:8], y[:8], rate)
    expected_updated = network(expected["updated"])
    errors = {"forward_max_abs_error": float(np.max(np.abs(predicted - array(expected["prediction"])))),
              "loss_abs_error": abs(loss - expected["loss"]),
              "gradient_max_abs_error": max(float(np.max(np.abs(g - array(e))))
                  for grad, ew, eb in zip(grads, expected["gradient_weights"], expected["gradient_bias"])
                  for g, e in zip(grad, (ew, eb))),
              "update_max_abs_error": max(float(np.max(np.abs(a[key] - b[key])))
                  for a, b in zip(updated, expected_updated) for key in ("weights", "bias"))}
    errors["tolerance"] = 1e-10
    errors["passed"] = all(v <= 1e-10 for k, v in errors.items() if k.endswith("error"))
    results = args.root / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "parity.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
    if not errors["passed"]:
        raise AssertionError("Rust/NumPy parity failed: " + json.dumps(errors))
    print("Rust/NumPy parity passed:", errors, flush=True)

    def run():
        net = copy.deepcopy(initial)
        start = time.perf_counter()
        for order in orders:
            for offset in range(0, len(order), batch):
                index = order[offset:offset + batch]
                update(net, x[index], y[index], rate)
        seconds = time.perf_counter() - start
        prediction = forward(net, x)[0][-1]
        return seconds, float(np.mean((prediction - y) ** 2))

    run()
    trials = []
    for i in range(1, 6):
        seconds, final_loss = run()
        trials.append({"trial": i, "seconds": seconds, "final_loss": final_loss})
        print(f"NumPy trial {i}: {seconds:.4f}s", flush=True)
    with (results / "benchmark_numpy.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["trial", "seconds", "final_loss"])
        writer.writeheader()
        writer.writerows(trials)
    blas = getattr(np.__config__, "CONFIG", {}).get("Build Dependencies", {}).get("blas", {})
    summary = {"implementation": "Python NumPy", "epochs": fixture["epochs"], "samples": len(x),
               "batch_size": batch, "learning_rate": rate, "dtype": "float64", "warmups": 1,
               "repetitions": 5, "threads": 1, "backend": {"name": blas.get("name", "unknown"),
                   "version": blas.get("version", "unknown")},
               "median_seconds": statistics.median(r["seconds"] for r in trials), "trials": trials,
               "environment": {"python": platform.python_version(), "numpy": np.__version__,
                   "os": platform.system() + " " + platform.release(), "architecture": platform.machine(),
                   "cpu": os.environ.get("PROCESSOR_IDENTIFIER", platform.processor()),
                   "thread_environment": {k: os.environ[k] for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}}
    (results / "benchmark_numpy.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rust = json.loads((results / "benchmark_rust.json").read_text(encoding="utf-8"))
    final_difference = abs(rust["trials"][0]["final_loss"] - trials[0]["final_loss"])
    if final_difference > 1e-9:
        raise AssertionError(f"Final training loss mismatch: {final_difference}")
    print(f"Median: Rust {rust['median_seconds']:.4f}s, NumPy {summary['median_seconds']:.4f}s", flush=True)


if __name__ == "__main__":
    main()
