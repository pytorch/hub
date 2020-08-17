#!/bin/bash
set -e
. ~/miniconda3/etc/profile.d/conda.sh
conda activate base

ALL_FILE=$(find *.md ! -name README.md)
TEMP_PY="temp.py"
CUDAS="nvidia"

for f in $ALL_FILE
do
  echo "Running pytorch example in $f"
  # FIXME: NVIDIA models checkoints are on cuda
  if [[ $f = $CUDAS* ]]; then
    echo "...skipped due to cuda checkpoints."
  elif [[ $f = "pytorch_fairseq_translation"* ]]; then
    echo "...temporarily disabled"
  else
    sed -n '/^```python/,/^```/ p' < $f | sed '/^```/ d' > $TEMP_PY
    python $TEMP_PY

    if [ -f "$TEMP_PY" ]; then
      rm $TEMP_PY
    fi
  fi
done
