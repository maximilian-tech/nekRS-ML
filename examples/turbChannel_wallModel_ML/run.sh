#!/bin/env bash
set -x

. $HOME/spack/share/spack/setup-env.sh
spack env activate gcc15-tc

set -euo pipefail

PREFIX=/data/horse/ws/s0872522-testing/nekRS-ML

export NEKRS_HOME=$PREFIX/nekrs_install
export PYTHONPATH=$NEKRS_HOME/3rd_party/rdq/:.


LAUNCHER=mpiexec # srun not supported on ROMEO

$LAUNCHER -n 4 -- $NEKRS_HOME/bin/nekrs --setup turbChannel_train : \
       -n 1 -- python3 ./torch_dataloader_consumer.py \
       | tee log.out
