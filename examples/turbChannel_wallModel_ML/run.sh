#!/bin/env bash

set -euo pipefail

. $HOME/spack/share/spack/setup-env.sh
spack env activate gcc15-tc

export NEKRS_HOME=$HOME/.local/nekrs
export PYTHONPATH=$HOME/repos/distributed-data-queue/_build/install/:.

mpirun -np 4 -- $NEKRS_HOME/bin/nekrs --setup turbChannel_train : \
       -np 1 -- python3 ./torch_dataloader_consumer.py \
       | tee log.out