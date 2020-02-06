#!/bin/bash
export PATH="/afs/cern.ch/work/a/asogaard/public/miniconda2/bin:$PATH"
cd /afs/cern.ch/work/e/ehansen/public/adversarial/
source activate.sh
python -m prepro.converterDJR
