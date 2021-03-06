#!/bin/bash
#PBS -N myFirstJob
#PBS -q gpu -l select=1:mem=20gb:ngpus=1:scratch_local=3gb:cl_adan=True 
#PBS -m ae
# The 4 lines above are options for scheduling system: job will run 1 hour at maximum, 1 machine with 4 processors + 4gb RAM memory + 10gb scratch memory are requested, email notification will be sent when the job aborts (a) or ends (e)

# define a DATADIR variable: directory where the input files are taken from and where output will be copied to
DATADIR=/storage/brno6/home/jakubsekula/

# append a line to a file "jobs_info.txt" containing the ID of the job, the hostname of node it is run on and the path to a scratch directory
# this information helps to find a scratch directory in case the job fails and you need to remove the scratch directory manually
echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs/jobs_info.txt

# test if scratch directory is set
# if scratch directory is not set, issue error message and exit
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

# copy input file "h2o.com" to scratch directory
# if the copy operation fails, issue error message and exit
#cp -r /storage/brno6/home/jakubsekula/HWR.2021-03-29/  $SCRATCHDIR || { echo >&2 "Error while copying input file(s)!"; exit 2; }

$DATADIR/train_specific.sh

#clean the SCRATCH directory
clean_scratch
