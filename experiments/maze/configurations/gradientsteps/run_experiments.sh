#!/bin/bash
#export PYTHONPATH="${PYTHONPATH}:/AIDM_B_milestone_DQN/"
RANDOM=$$
for rep in 1 2 3 4 5
do
    for value in 16 64 128
    do
      cp experiments/maze/configurations/gradient_steps/15x15.yaml config.yaml
            head -n -2 config.yaml > temp.txt ; mv temp.txt config.yaml
      echo "seed: $RANDOM" >> config.yaml
      echo "gradient_steps: $value" >> config.yaml
      python3 experiments/maze/main.py -c config.yaml
    done
done
rm config.yaml