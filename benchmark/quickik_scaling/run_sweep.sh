#!/usr/bin/env bash
# Sweeps quickik-scaling across 1/2/4/8/16 threads via taskset -c, printing one
# line per thread count and accumulating each run's data point into
# ../plot/results/quickik-scaling.json. Run from anywhere; builds in release
# mode first. Chart it afterwards with `python ../plot/plot_scaling.py`.
set -euo pipefail

cd "$(dirname "$0")/../.."
cargo build --release -p quickik-scaling

BIN=target/release/quickik-scaling
for cpus in 0 0-1 0-3 0-7 0-15; do
    taskset -c "$cpus" "$BIN"
done
