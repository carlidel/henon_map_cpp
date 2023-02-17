import itertools

omega_list = [(0.168, 0.201, 0.75), (0.310, 0.320, 0.25)]

epsilon_list = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]

mu_list = [0.0, 0.25, 0.5, 0.75, 1.0]

# Create a list of all combinations of the above parameters
param_list = list(itertools.product(omega_list, epsilon_list, mu_list))

# Write the list of parameters to a file
with open("configs/all_jobs.txt", "w") as f:
    for (omega_x, omega_y, max_radius), epsilon, mu in param_list:
        f.write(f"{omega_x} {omega_y} {epsilon} {mu} {max_radius}\n")
