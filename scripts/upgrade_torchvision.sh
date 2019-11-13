#!/bin/bash

###### Usage #####
# In hub/ directory, run:
#   ./scripts/upgrade_torchvision.sh <target_version>
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
sed -i "s/load('pytorch\/vision[^']*'/load('pytorch\/vision:$1'/g" $DIR/../*.md
