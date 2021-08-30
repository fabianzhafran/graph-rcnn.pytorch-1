#!/bin/bash -l

#$ -l h_rt=24:00:00   # Specify the hard time limit for the job
#$ -N merge_flickr30k # Give job a name
#$ -j y               # Merge the error and output streams into a single file
#$ -V

python merge_graphs_flickr30k.py
