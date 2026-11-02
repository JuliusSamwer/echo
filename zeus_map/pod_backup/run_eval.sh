#!/usr/bin/env bash
set -x
cd /workspace/HWM_PLDM
source /workspace/HWM_PLDM/activate_env.sh
bash eval.sh "$1"
