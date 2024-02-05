#!/bin/bash
source /data/users4/vitkyal/softwares/miniconda3/bin/activate
source activate meshnet
echo $CONDA_DEFAULT_ENV
/data/users4/vitkyal/softwares/miniconda3/envs/meshnet/bin/python curriculum_training.py