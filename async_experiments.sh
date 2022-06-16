#source conda activate afidm
printf "terminating all previously running experiments .."
#screen -XS buffer quit; screen -XS denom quit; screen -XS maxtime quit; screen -XS grad quit

#printf "\nremoving results folder..\n"
#rm -rf experiments/maze/results

# --- 128x128x128 adaptive4 --
export PYTHONPATH="${PYTHONPATH}:/AIDM_B_milestone_DQN/"
screen -S aa -d -m bash -c './experiments/maze/configurations/architecture/run_experiments.sh 128 adaptive4; exec bash'

# --- 128x128x128 trad ---
export PYTHONPATH="${PYTHONPATH}:/AIDM_B_milestone_DQN/"
screen -S ab -d -m bash -c './experiments/maze/configurations/architecture/run_experiments.sh 128 traditional_milestones; exec bash'

# --- 256x256x256 adaptive4---
#export PYTHONPATH="${PYTHONPATH}:/AIDM_B_milestone_DQN/"
#screen -S ba -d -m bash -c './experiments/maze/configurations/architecture/run_experiments.sh 256 adaptive4; exec bash'

# --- 256x256x256 trad ---
#export PYTHONPATH="${PYTHONPATH}:/AIDM_B_milestone_DQN/"
#screen -S bb -d -m bash -c './experiments/maze/configurations/architecture/run_experiments.sh 256 traditional_milestones; exec bash'

printf "\n--- Started screen shells with experiments, they will not terminate automatically (because of ending with exec bash) --- \n\n"
screen -list
printf "\n--- to terminate all sessions, execute:     screen -XS aa quit; screen -XS ab quit; screen -XS ba quit; screen -XS bb quit\n"
