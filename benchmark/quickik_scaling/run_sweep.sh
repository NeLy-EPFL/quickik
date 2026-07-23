#!/usr/bin/env bash
# Sweeps quickik-scaling across 1/2/4/8/16 workers (passed explicitly via
# ParallelSolveConfig::n_workers -- the binary sweeps them itself in one run,
# no taskset needed), printing one line per worker count and writing every
# data point into ../plot/results/quickik-scaling.json. Run from anywhere;
# builds in release mode first. Chart it afterwards with
# `python ../plot/plot_scaling.py`.
set -euo pipefail

cd "$(dirname "$0")/../.."
cargo build --release -p quickik-scaling
target/release/quickik-scaling
