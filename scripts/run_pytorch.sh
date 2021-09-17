#!/bin/bash
. ~/miniconda3/etc/profile.d/conda.sh
conda activate base

ALL_FILES=$(find *.md ! -name README.md)
PYTHON_CODE_DIR="python_code"

mkdir $PYTHON_CODE_DIR

# Quick rundown: for each file we extract the python code that's within
# the ``` markers and we put that code in a corresponding .py file in $PYTHON_CODE_DIR
# Then we execute each of these python files with pytest in test_run_python_code.py
for f in $ALL_FILES
do
  f_no_ext=${f%.md}  # remove .md extension
  out_py=$PYTHON_CODE_DIR/$f_no_ext.py
  echo "Extracting Python code from $f into $out_py"
  sed -n '/^```python/,/^```/ p' < $f | sed '/^```/ d' > $out_py
done

pytest --junitxml=test-results/junit.xml test_run_python_code.py -vv
