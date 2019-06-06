#!/bin/bash
set -e
. ~/miniconda3/etc/profile.d/conda.sh
conda activate base

ALL_FILE=$(find *.md ! -name README.md)
TEMP_PY="temp.py"
GANs="facebookresearch_pytorch-gan-zoo"

for f in $ALL_FILE
do
  echo "Running pytorch example in $f"
  # FIXME: GAN models checkpoints are on cuda.
  if [[ $f = $GANs* ]]; then
    echo "...skipped due to cuda checkpoints."
  else
    sed -n '/^```python/,/^```/ p' < $f | sed '/^```/ d' > $TEMP_PY
    python $TEMP_PY

    if [ -f "$TEMP_PY" ]; then
      rm $TEMP_PY
    fi
  fi
done
