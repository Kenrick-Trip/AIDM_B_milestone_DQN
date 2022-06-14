#source conda activate afidm
rm experiments/maze/results
export PYTHONPATH="${PYTHONPATH}:/AIDM_B_milestone_DQN/"
screen -S buffer -d -m bash -c './experiments/maze/configurations/buffer/run_experiments.sh; exec bash'
export PYTHONPATH="${PYTHONPATH}:/AIDM_B_milestone_DQN/"
#screen -S denominator -d -m bash -c './experiments/maze/configurations/denominator/run_experiments.sh; exec bash'
screen -S maxtimesteps -d -m bash -c './experiments/maze/configurations/maxtimesteps/run_experiments.sh; exec bash'
#screen -S gradientsteps -d -m bash -c './experiments/maze/configurations/gradientsteps/run_experiments.sh; exec bash'

echo "\n--- Started screen shells with experiments, they will terminate if completed --- \n"
screen -list
