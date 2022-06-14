#!/bin/bash
#export PYTHONPATH="${PYTHONPATH}:/AIDM_B_milestone_DQN/"
RANDOM=$$
for rep in 1 2 3 4 5
do
    for value in "maze-custom-15x15-v1-500" "maze-custom-15x15-v1-800" "maze-custom-15x15-v1-1100"
    do
      cp experiments/maze/configurations/maxtimesteps/15x15.yaml config_steps.yaml
            head -n -2 config_steps.yaml > temp_steps.txt ; mv temp_steps.txt config_steps.yaml
      echo "seed: $RANDOM" >> config_steps.yaml
      echo "env: '$value'" >> config_steps.yaml
      python3 experiments/maze/main.py -c config_steps.yaml
    done
done
rm config_steps.yaml