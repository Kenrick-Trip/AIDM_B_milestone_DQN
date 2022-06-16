#!/bin/bash
#export PYTHONPATH="${PYTHONPATH}:/AIDM_B_milestone_DQN/"
if [ -z "$1" ]
  then
    echo "No architecture supplied"
    exit 0
fi

if [ -z "$2" ]
  then
    echo "No exploration method provided"
    exit 0
fi

if [ -z "$3" ]
  then
    echo "No Num milestones provided"
    exit 0
fi

RANDOM=$$
for rep in 1 2 3 4 5
do
      cp experiments/maze/configurations/architecture/15x15.yaml "/tmp/config_$1_$2.yaml"
            head -n -5 "/tmp/config_$1_$2.yaml" > "/tmp/temp_$1_$2.txt" ; mv "/tmp/temp_$1_$2.txt" "/tmp/config_$1_$2.yaml"
      echo -e "  net_arch: [$1, $1, $1]" >> "/tmp/config_$1_$2.yaml"
      echo "seed: $RANDOM" >> "/tmp/config_$1_$2.yaml"
      echo "exploration_method: '$2'" >> "/tmp/config_$1_$2.yaml"
      echo "results_folder: '$1_$2'" >> "/tmp/config_$1_$2.yaml"
      echo "num_milestones: $3" >> "/tmp/config_$1_$2.yaml"
      python3 experiments/maze/main.py -c "/tmp/config_$1_$2.yaml"
done
#rm "/tmp/config_$1_$2.yaml"
