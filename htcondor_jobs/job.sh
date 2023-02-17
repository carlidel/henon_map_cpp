#!/bin/bash

source /cvmfs/sft.cern.ch/lcg/views/LCG_102cuda/x86_64-centos7-gcc8-opt/setup.sh
source /afs/cern.ch/work/c/camontan/public/tangent_map/myenv/bin/activate
export EOS_MGM_URL=root://eosuser.cern.ch

which python

# echo the 4 arguments received by the script
echo $1
echo $2
echo $3
echo $4
echo $5

# run the simulation
python3 /afs/cern.ch/work/c/camontan/public/henon_map_cpp/htcondor_jobs/run_sim.py --omegax $1 --omegay $2 --epsilon $4 --mu $5 --max_radius $3

# copy the output to eos
eos cp *.h5 /eos/user/c/camontan/lhc_dynamic_data

# remove the output
rm *.h5
