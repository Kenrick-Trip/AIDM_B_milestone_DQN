#!/bin/bash
#export PYTHONPATH="${PYTHONPATH}:/AIDM_B_milestone_DQN/"
RANDOM=$$
for rep in 1 2 3 4 5
do
    for value in 2000 3000 4000
    do
      cp experiments/maze/configurations/denominator/15x15.yaml config_denom.yaml
            head -n -2 config_denom.yaml > temp_denom.txt ; mv temp_denom.txt config_denom.yaml
      echo "seed: $RANDOM" >> config_denom.yaml
      echo "denominator: $value" >> config_denom.yaml
      python3 experiments/maze/main.py -c config_denom.yaml
    done
done
rm config_denom.yaml