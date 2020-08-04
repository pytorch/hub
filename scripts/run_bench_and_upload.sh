#!/bin/bash
set -e
. ~/miniconda3/etc/profile.d/conda.sh
conda activate base

BENCHMARK_DATA=".benchmarks"
mkdir -p ${BENCHMARK_DATA}
pytest benchmark/test_bench.py --benchmark-sort=Name --benchmark-json=${BENCHMARK_DATA}/hub.json
python benchmark/upload_scribe.py --pytest_bench_json ${BENCHMARK_DATA}/hub.json
