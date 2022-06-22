import numpy as np
import matplotlib.pyplot as plt

directory = "experiments/mountaincar/results/results_mc_run_bas2"

results_adapt4 = [
    "2022-06-22-t-102700",
    "2022-06-22-t-102610",
    "2022-06-22-t-102449"
]

results_traditional = [
    "2022-06-22-t-102701",
    "2022-06-22-t-102608",
    "2022-06-22-t-102447"
]

fig, ax = plt.subplots()

init_data = np.load("../{}/{}/evaluations.npz".format(directory, results_traditional[2]))
x = init_data["timesteps"]

n = 124
print(n)
y_adapt4 = np.zeros([len(results_adapt4), n])
y_traditional = np.zeros([len(results_adapt4), n])

mean_adapt4 = np.zeros([n])
std_adapt4 = np.zeros([n])
mean_traditional = np.zeros([n])
std_traditional = np.zeros([n])

std_max_adapt = np.zeros([n])
std_min_adapt = np.zeros([n])
std_max_traditional = np.zeros([n])
std_min_traditional = np.zeros([n])

for i in range(len(results_adapt4)):
    data = np.load("../{}/{}/evaluations.npz".format(directory, results_adapt4[i]))
    y_adapt4[i] = data["results"][0:124, 0]
    data = np.load("../{}/{}/evaluations.npz".format(directory, results_traditional[i]))
    y_traditional[i] = data["results"][0:124, 0]

for k in range(n):
    mean_adapt4[k] = np.mean(y_adapt4[:, k])
    std_adapt4[k] = np.std(y_adapt4[:, k])

    if mean_adapt4[k] + std_adapt4[k] >= 6:
        std_max_adapt[k] = 6
    else:
        std_max_adapt[k] = mean_adapt4[k] + std_adapt4[k]

    if mean_adapt4[k] - std_adapt4[k] <= 0:
        std_min_adapt[k] = 0
    else:
        std_min_adapt[k] = mean_adapt4[k] - std_adapt4[k]

    mean_traditional[k] = np.mean(y_traditional[:, k])
    std_traditional[k] = np.std(y_traditional[:, k])

    if mean_traditional[k] + std_traditional[k] >= 6:
        std_max_traditional[k] = 6
    else:
        std_max_traditional[k] = mean_traditional[k] + std_traditional[k]

    if mean_traditional[k] - std_traditional[k] <= 0:
        std_min_traditional[k] = 0
    else:
        std_min_traditional[k] = mean_traditional[k] - std_traditional[k]


ax.plot(x, mean_adapt4, color='orange', alpha=0.5, label="adaptive")
ax.fill_between(x, std_min_adapt, std_max_adapt, alpha=0.15, color="orange")
ax.plot(x, mean_traditional, color='blue', alpha=0.5, label="traditional_milestones")
ax.fill_between(x, std_min_traditional, std_max_traditional, alpha=0.15, color="blue")
ax.plot(x, 6*np.ones([n]), color="cyan", alpha=0.5, label="max_reward(6.0)")

ax.set_ylabel("Reward")
ax.set_xlabel("Timestep")
plt.legend(loc=4)
plt.show()

