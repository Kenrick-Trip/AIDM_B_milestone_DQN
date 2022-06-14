#!/bin/bash
#export PYTHONPATH="${PYTHONPATH}:/AIDM_B_milestone_DQN/"
RANDOM=$$
for rep in 1 2 3 4 5
do
    for value in 1400000 600000 1000000
    do
      cp experiments/maze/configurations/buffer/15x15.yaml config_buffer.yaml
            head -n -2 config_buffer.yaml > temp_buffer.txt ; mv temp_buffer.txt config_buffer.yaml
      echo "seed: $RANDOM" >> config_buffer.yaml
      echo "buffer_size: $value" >> config_buffer.yaml
      python3 experiments/maze/main.py -c config_buffer.yaml
    done
done
rm config_buffer.yaml
