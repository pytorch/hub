#!/bin/bash
set -e
. ~/miniconda3/etc/profile.d/conda.sh
conda activate base

BENCHMARK_DATA=".benchmarks"
mkdir -p ${BENCHMARK_DATA}
pytest benchmark/test_bench.py --benchmark-sort=Name --benchmark-json=${BENCHMARK_DATA}/hub.json

# Token is only present for certain jobs, only upload if present
if [ -z "$SCRIBE_GRAPHQL_ACCESS_TOKEN" ]
then
    echo "Skipping benchmark upload, token is missing."
else
    python benchmark/upload_scribe.py --pytest_bench_json ${BENCHMARK_DATA}/hub.json
fi
