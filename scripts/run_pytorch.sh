#!/bin/bash
set -e
. ~/miniconda3/etc/profile.d/conda.sh
conda activate base

ALL_FILE=$(find *.md ! -name README.md)
TEMP_PY="temp.py"

for f in $ALL_FILE
do
  echo "Running pytorch example in $f"
  sed -n '/^```python/,/^```/ p' < $f | sed '/^```/ d' > $TEMP_PY
  python $TEMP_PY

  if [ -f "$TEMP_PY" ]; then
    rm $TEMP_PY
  fi
done
