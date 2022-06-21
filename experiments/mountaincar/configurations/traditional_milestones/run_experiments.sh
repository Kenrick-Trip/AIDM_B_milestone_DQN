#!/bin/bash
RANDOM=$$
for rep in 1 2 3 4 5
do
      echo "seed: $RANDOM" >> "experiments/mountaincar/configurations/traditional_milestones/MCtraditionalmilestones.yaml"
      python3 experiments/mountaincar/main.py -c "experiments/mountaincar/configurations/traditional_milestones/MCtraditionalmilestones.yaml"
done
