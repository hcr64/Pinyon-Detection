#!/bin/bash
JOB_ID=$1

echo ***DATE***
srun date
echo 

echo ***OUPUT***
cat /scratch/hcr64/$JOB_ID.txt
echo

echo ***ERRORS***
cat /scratch/hcr64/$JOB_ID.err
echo