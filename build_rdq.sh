#!/bin/env bash

set -euo pipefail

CWD=$PWD

rm -rf $PWD/_build_rdq
mkdir $PWD/_build_rdq
cd $PWD/_build_rdq
# NEKRS_INSTALL_DIR
SCOREP_WRAPPER=off cmake ../3rd_party/remote-data-queue  \
	--fresh \
	-DCMAKE_BUILD_TYPE= \
	-DCMAKE_INSTALL_PREFIX=${NEKRS_HOME:?}/3rd_party/rdq \
	-DRDQ_ENABLE_PYTHON=1 \
#	-DRDQ_ENABLE_BENCHMARKS=1 \
#	-DCMAKE_C_COMPILER=scorep-icx \
#	-DMPI_C_COMPILER=scorep-mpiicc
	
make -B  -j
make install
