#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:/AIDM_B_milestone_DQN/"
for seed in 1 2 3 4 5 6 7 8 9
do
    for value in traditional traditional_milestones adaptive1 adaptive2 deep_exploration
    do
	cp experiments/mountaincar/mountaincar.yaml config.yaml
        head -n -4 config.yaml > temp.txt ; mv temp.txt config.yaml
	echo "demosteps: 0" >> config.yaml
	echo "trainsteps: 1000000" >> config.yaml
        echo "seed: $seed" >> config.yaml
	echo "exploration_method: '$value'" >> config.yaml
	python3 experiments/mountaincar/main.py -c config.yaml
    done
done
rm config.yaml
zip -r results2.zip results/.
#head -n -1 AIDM_B_milestone_DQN/experiments/mountaincar/mountaincar.yaml > temp.txt
