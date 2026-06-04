#!/bin/bash

# find all jobs, only main ones thoughh so exclude date and things like that
sacct -u hcr64 --starttime=today --format=JobID,JobName,State,Elapsed,End | grep pinyon_m
