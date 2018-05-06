#!/bin/bash
#$ -N t0.5-r1000-lsi-50-custom-stop-no-vote
#$ -q seal
#$ -m beas

PROJ_PATH=/share/seal/junwel1/icst2017/new_data
T_RATIO=0.5
R_TIMES=1000
T_NAME=lsi-50-custom-stop-no-vote

cd $PROJ_PATH

module load anaconda/3.6-5.0.1
source activate ityc-ext

echo "$(date): Script Starts"

python train_and_test.py $T_RATIO $R_TIMES $T_NAME

echo "$(date): Script Ends."
