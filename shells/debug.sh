#!/bin/bash
echo ***DATE***
srun date
echo 

echo ***OUPUT***
cat /scratch/hcr64/pinyon_output.txt
echo

echo ***ERRORS***
cat /scratch/hcr64/pinyon_output.err
echo