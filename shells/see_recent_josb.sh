#!/bin/bash

# find all jobs, only main ones thoughh so exclude date and things like that

# today
echo "********** Today **********"
sacct -u hcr64 --starttime=today --format=JobID,JobName,State,Elapsed,End | grep pinyon_m
echo 

# jobs from yesterday
echo "********** Yesterday **********"
sacct -u hcr64 --starttime=$(date -d yesterday +%Y-%m-%d) --endtime=$(date +%Y-%m-%d) --format=JobID,JobName,State,Elapsed,End | grep pinyon_m
echo