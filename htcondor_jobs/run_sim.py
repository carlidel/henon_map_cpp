import argparse
import datetime

import h5py
import numpy as np

import henon_map_cpp as hm


def f_to_str(f):
    # Convert a float to a string, but replace the decimal point with a "d"
    return str(f).replace(".", "d")


parser = argparse.ArgumentParser()
parser.add_argument("--omegax", type=float, default=0.31)
parser.add_argument("--omegay", type=float, default=0.32)
parser.add_argument("--epsilon", type=float, default=0.0)
parser.add_argument("--mu", type=float, default=0.0)
parser.add_argument("--max_radius", type=float, default=0.25)
parser.add_argument("--num_points", type=int, default=100)
parser.add_argument("--num_iterations", type=int, default=int(1e8))

args = parser.parse_args()

# Create the initial conditions
positions = np.linspace(0, args.max_radius, args.num_points)

xx, yy = np.meshgrid(positions, positions)

x = xx.flatten()
px = np.zeros_like(x)
y = yy.flatten()
py = np.zeros_like(y)

particles = hm.particles(x, px, y, py)

# start chronometer
start = datetime.datetime.now()

tracker = hm.henon_tracker(
    args.num_iterations, args.omegax, args.omegay, "sps", epsilon=args.epsilon
)
tracker.track(particles, args.num_iterations, mu=args.mu, barrier=10.0)
steps = particles.get_steps()

# end chronometer
end = datetime.datetime.now()

# print elapsed time
print("Elapsed time: {}".format(end - start))


# Save the results
with h5py.File(
    f"stability_henon_{f_to_str(args.omegax)}_{f_to_str(args.omegay)}_{f_to_str(args.epsilon)}_{f_to_str(args.mu)}_{f_to_str(args.max_radius)}.h5",
    "w",
) as f:
    f.create_dataset("steps", data=steps)
