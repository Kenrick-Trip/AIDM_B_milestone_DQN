#!/bin/bash
#export PYTHONPATH="${PYTHONPATH}:/AIDM_B_milestone_DQN/"
RANDOM=$$
for rep in 1 2 3 4 5
do
    for value in 2000 3000 4000
    do
      cp experiments/maze/configurations/denominator/15x15.yaml config.yaml
            head -n -2 config.yaml > temp.txt ; mv temp.txt config.yaml
      echo "seed: $RANDOM" >> config.yaml
      echo "denominator: $value" >> config.yaml
      python3 experiments/maze/main.py -c config.yaml
    done
done
rm config.yaml