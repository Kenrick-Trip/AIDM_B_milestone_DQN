export PYTHONPATH="${PYTHONPATH}:/AIDM_B_milestone_DQN/"
screen -S aa -d -m bash -c './experiments/mountaincar/configurations/adaptive4/run_experiments.sh; exec bash'
screen -S ab -d -m bash -c './experiments/mountaincar/configurations/traditional_milestones/run_experiments.sh; exec bash'
