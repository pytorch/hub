#!/bin/bash

pytest test_bench.py -k cuda-jit --fuser legacy --benchmark-json legacy.json
pytest test_bench.py -k cuda-jit --fuser te --benchmark-json te.json
python compare.py legacy.json te.json
