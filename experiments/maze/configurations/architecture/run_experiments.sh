#!/bin/bash
#export PYTHONPATH="${PYTHONPATH}:/AIDM_B_milestone_DQN/"
if [ -z "$1" ]
  then
    echo "No architecture supplied"
    exit 0
fi

RANDOM=$$
for rep in 1 2 3 4 5
do
    for value in 'adaptive4' 'traditional_milestones'
    do
      cp experiments/maze/configurations/architecture/15x15.yaml config_arch.yaml
            head -n -2 config_arch.yaml > temp_arch.txt ; mv temp_arch.txt config_arch.yaml
      echo -e "  net_arch: [$1, $1, $1]" >> config_arch.yaml
      echo "seed: $RANDOM" >> config_arch.yaml
      echo "exploration_method: $value" >> config_arch.yaml
      python3 experiments/maze/main.py -c config_arch.yaml
    done
done
rm config_arch.yaml
