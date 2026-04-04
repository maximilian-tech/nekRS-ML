#!/bin/env bash
set -x
set -euo pipefail

. $HOME/spack/share/spack/setup-env.sh
spack env activate gcc15-tc

PREFIX=/data/horse/ws/s0872522-testing/

export NEKRS_HOME=$PREFIX/nekrs_install
export PYTHONPATH=$PREFIX/remote-data-queue/_build/install/:.

mpirun -np 4 -- $NEKRS_HOME/bin/nekrs --setup turbChannel_train : \
       -np 1 -- python3 ./torch_dataloader_consumer.py \
       | tee log.out
