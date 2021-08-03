#!/bin/bash
set -e
. ~/miniconda3/etc/profile.d/conda.sh
conda activate base

ALL_FILES=$(find *.md ! -name README.md)
TEMP_PY="temp.py"
PYTHON_CODE_DIR="python_code"
CUDAS="nvidia"

mkdir $PYTHON_CODE_DIR

for f in $ALL_FILES
do
  f_no_ext=${f%.md}  # remove .md extension
  out_py=$PYTHON_CODE_DIR/$f_no_ext.py
  echo "Extracting Python code from $f into $out_py"
  sed -n '/^```python/,/^```/ p' < $f | sed '/^```/ d' > $out_py
done

pytest -v -s test_run_python_code.py
rm -r $PYTHON_CODE_DIR
