#!/usr/bin/env python3
"""Compare C++ reducer against the actual training Python methods.
Usage: python3 test_policy_health_parity.py /path/to/flightpy/src/PAMPC
Requires numpy and g++; no ROS runtime or SVO bindings required.
"""
import ast
import __future__
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np


def method(path, name):
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace[name]


def main():
    training = Path(sys.argv[1])
    raw = method(training / "shadow/rpg_svo_shadow.py", "_perception_health")
    normalize = method(training / "pampc_state_env.py", "_set_svo_health")
    state = SimpleNamespace(_track_ages={}, _add_svo_health_observation=True,
        cfg={"state_estimation": {"sac_health_use_landmark_survival_ratio": True}},
        _svo_health=np.zeros((1, 4)))
    rng = np.random.default_rng(42)
    lines, expected = [], []
    for frame in range(200):
        if frame % 40 == 0:
            state._track_ages.clear()
            lines.append("-1 0 0")
        n = int(rng.integers(0, 250))
        tracked = rng.random(n) < .8
        ids = rng.integers(-1, 100, n)
        errors = rng.uniform(0, 6, n)
        errors[rng.random(n) < .3] = np.nan
        health = raw(state, {"is_tracked": tracked, "track_ids": ids,
                             "reprojection_errors_px": errors})
        normalize(state, 0, dict(health, telemetry_available=True))
        expected.append(state._svo_health[0].copy())
        unique_ids = sorted(set(int(x) for x in ids[tracked] if x >= 0))
        finite_errors = errors[tracked & np.isfinite(errors)]
        lines.append(f"{tracked.sum()} {len(unique_ids)} {len(finite_errors)} " +
                     " ".join(map(str, unique_ids)) + " " + " ".join(map(str, finite_errors)))
    source = r"""
#include <svo_ros/policy_health.h>
#include <iostream>
#include <iomanip>
int main() {
  svo::PolicyHealthAccumulator h;
  int count, ni, ne;
  while (std::cin >> count >> ni >> ne) {
    if (count < 0) { h.reset(); continue; }
    std::unordered_set<int64_t> ids;
    for (int i=0; i<ni; ++i) { int64_t id; std::cin >> id; ids.insert(id); }
    std::vector<double> errors(ne);
    for (auto& e : errors) std::cin >> e;
    const auto v = h.update(count, ids, errors);
    for (double x : v.normalized) std::cout << std::setprecision(17) << x << " ";
    std::cout << "\n";
  }
}
"""
    with tempfile.TemporaryDirectory() as tmp:
        cpp = Path(tmp) / "parity.cpp"
        exe = Path(tmp) / "parity"
        cpp.write_text(source)
        include = Path(__file__).resolve().parents[1] / "include"
        subprocess.run(["g++", "-std=c++11", "-Wall", "-Wextra", "-Werror",
                        "-I", str(include), str(cpp), "-o", str(exe)], check=True)
        output = subprocess.check_output([str(exe)], input="\n".join(lines), text=True)
    actual = np.fromstring(output, sep=" ").reshape(-1, 4)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
    print("PASS: 200 frames match training health and normalization, including resets")


if __name__ == "__main__":
    main()
