#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:/AIDM_B_milestone_DQN/"
RANDOM=$$
for rep in 1 2 3
do
#    for value in traditional traditional_milestones adaptive1 adaptive2 deep_exploration
    for value in traditional_milestones adaptive3 adaptive4
    do
      cp experiments/maze/maze.yaml config.yaml
            head -n -4 config.yaml > temp.txt ; mv temp.txt config.yaml
      echo "demosteps: 0" >> config.yaml
      echo "trainsteps: 200000" >> config.yaml
            echo "seed: $RANDOM" >> config.yaml
      echo "exploration_method: '$value'" >> config.yaml
      python3 experiments/maze/main.py -c config.yaml
    done
done
rm config.yaml
zip -r results.zip experiments/maze/results_benchmark1/.
#head -n -1 AIDM_B_milestone_DQN/experiments/mountaincar/mountaincar.yaml > temp.txt
