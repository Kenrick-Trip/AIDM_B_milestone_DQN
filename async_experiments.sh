#source conda activate afidm
printf "terminating all previously running experiments .."
screen -XS buffer quit; screen -XS denom quit; screen -XS maxtime quit; screen -XS grad quit

printf "\nremoving results folder..\n"
rm -rf experiments/maze/results

# --- buffer ---
export PYTHONPATH="${PYTHONPATH}:/AIDM_B_milestone_DQN/"
screen -S buffer -d -m bash -c './experiments/maze/configurations/buffer/run_experiments.sh; exec bash'

# --- denominator ---
export PYTHONPATH="${PYTHONPATH}:/AIDM_B_milestone_DQN/"
screen -S denominator -d -m bash -c './experiments/maze/configurations/denominator/run_experiments.sh; exec bash'

# --- maxtimesteps ---
export PYTHONPATH="${PYTHONPATH}:/AIDM_B_milestone_DQN/"
screen -S maxtimesteps -d -m bash -c './experiments/maze/configurations/maxtimesteps/run_experiments.sh; exec bash'

# --- gradientsteps ---
export PYTHONPATH="${PYTHONPATH}:/AIDM_B_milestone_DQN/"
screen -S gradientsteps -d -m bash -c './experiments/maze/configurations/gradientsteps/run_experiments.sh; exec bash'

printf "\n--- Started screen shells with experiments, they will not terminate automatically (because of ending with exec bash) --- \n\n"
screen -list
printf "\n--- to terminate all sessions, execute:     screen -XS buffer quit; screen -XS denom quit; screen -XS maxtime quit; screen -XS grad quit\n"