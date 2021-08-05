#!/bin/bash
. ~/miniconda3/etc/profile.d/conda.sh
conda activate base

ALL_FILE=$(find *.md ! -name README.md)
TEMP_PY="temp.py"
CUDAS="nvidia"

declare -i error_code=0

for f in $ALL_FILE
do
  echo "Running pytorch example in $f"
  # FIXME: NVIDIA models checkoints are on cuda
  if [[ $f = $CUDAS* ]]; then
    echo "...skipped due to cuda checkpoints."
  elif [[ $f = "pytorch_fairseq_translation"* ]]; then
    echo "...temporarily disabled"
  # FIXME: torch.nn.modules.module.ModuleAttributeError: 'autoShape' object has no attribute 'fuse'
  elif [[ $f = "ultralytics_yolov5"* ]]; then
    echo "...temporarily disabled"
  elif [[ $f = "huggingface_pytorch-transformers"* ]]; then
    echo "...temporarily disabled"
  # FIXME: TypeError: compose() got an unexpected keyword argument 'strict'
  elif [[ $f = "pytorch_fairseq_roberta"* ]]; then
    echo "...temporarily disabled"
  # FIXME: rate limiting
  elif [[ $f = "yanndubs_lossyless_clip_compressor"* ]]; then
    echo "...skipped to avoid downloading a dataset"
  # FIXME: avoid downloading dataset
  else
    sed -n '/^```python/,/^```/ p' < $f | sed '/^```/ d' > $TEMP_PY
    python $TEMP_PY
    error_code+=$?

    if [ -f "$TEMP_PY" ]; then
      rm $TEMP_PY
    fi
  fi
done

exit $error_code
