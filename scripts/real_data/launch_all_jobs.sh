#!/bin/bash

for SEED in {1..5}; do
    FIRST=$(qsub -v SEED=$SEED -N MDS_rao_$SEED -tc 1 scripts/torque_mds_rao.sh)
    SECOND=$(qsub -v SEED=$SEED -N PM2_rao_$SEED -tc 1 -hold_jid_ad MDS_rao_$SEED scripts/torque_pm2_rao.sh)
    SECOND=$(qsub -v SEED=$SEED -N NB02cst_rao_$SEED -tc 1 -hold_jid_ad MDS_rao_$SEED scripts/torque_nb02cst_rao.sh)
done;

for SEED in {1..1}; do
    SECOND=$(qsub -v SEED=$SEED -N NB2cst_rao_$SEED -tc 1 -hold_jid_ad MDS_rao_$SEED scripts/torque_nb2cst_rao.sh)
done;
