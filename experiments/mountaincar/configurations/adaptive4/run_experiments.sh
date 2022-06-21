#!/bin/bash
RANDOM=$$
for rep in 1 2 3 4 5
do
      echo "seed: $RANDOM" >> "experiments/mountaincar/configurations/adaptive4/MCadaptive4.yaml"
      python3 experiments/mountaincar/main.py -c "experiments/mountaincar/configurations/adaptive4/MCadaptive4.yaml"
done
