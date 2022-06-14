#!/bin/bash
#export PYTHONPATH="${PYTHONPATH}:/AIDM_B_milestone_DQN/"
RANDOM=$$
for rep in 1 2 3 4 5
do
    for value in 16 64 128
    do
      cp experiments/maze/configurations/gradientsteps/15x15.yaml config_grad.yaml
            head -n -2 config_grad.yaml > temp_grad.txt ; mv temp_grad.txt config_grad.yaml
      echo "seed: $RANDOM" >> config_grad.yaml
      echo "gradient_steps: $value" >> config_grad.yaml
      python3 experiments/maze/main.py -c config_grad.yaml
    done
done
rm config_grad.yaml