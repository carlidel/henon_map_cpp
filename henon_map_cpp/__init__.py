import numpy as np
import numba.cuda
import pandas as pd

#from .henon_map_engine import cpu_henon, gpu_henon, is_cuda_device_available

from .henon_map_engine import particles_4d, particles_4d_gpu, storage_4d, is_cuda_device_available
from .henon_map_engine import birkhoff_weights as cpp_birkhoff_weights
from .henon_map_engine import henon_tracker as cpp_henon_tracker
from .henon_map_engine import henon_tracker_gpu as cpp_henon_tracker_gpu

def gpu_available():
    return numba.cuda.is_available()

def birkhoff_weights(n):
    return np.asarray(cpp_birkhoff_weights(n))

class particles():
    def __init__(self, x_0, px_0, y_0, py_0, force_CPU=False):
        # check if the first 5 arguments are numpy arrays of the same 1D shape
        if not (isinstance(x_0, np.ndarray) and isinstance(px_0, np.ndarray) and
                isinstance(y_0, np.ndarray) and isinstance(py_0, np.ndarray)):
            raise TypeError("Arguments must be numpy arrays.")
        if not (x_0.ndim == 1 and px_0.ndim == 1 and y_0.ndim == 1 and
                py_0.ndim == 1):
            raise TypeError("Arguments must be 1D arrays.")
        if not (x_0.shape == px_0.shape == y_0.shape == py_0.shape):
            raise ValueError("Arguments must be of the same shape.")     
        
        # check if system has a nvidia gpu available
        GPU = gpu_available()
        if force_CPU or not GPU:
            self.particles = particles_4d(x_0, px_0, y_0, py_0)
        else:
            self.particles = particles_4d_gpu(x_0, px_0, y_0, py_0)
    
    def reset(self):
        self.particles.reset()

    def add_ghost(self, module, displacement_kind):
        self.particles.add_ghost(module, displacement_kind)

    def renormalize(self, module):
        self.particles.renormalize(module)

    def get_displacement_module(self):
        return np.asarray(self.particles.get_displacement_module())

    def get_displacement_direction(self):
        return np.asarray(self.particles.get_displacement_direction())

    def get_x(self):
        return np.asarray(self.particles.get_x())
    
    def get_px(self):
        return np.asarray(self.particles.get_px())
    
    def get_y(self):
        return np.asarray(self.particles.get_y())
    
    def get_py(self):
        return np.asarray(self.particles.get_py())

    def get_steps(self):
        return np.asarray(self.particles.get_steps())


class henon_tracker():
    def __init__(self, N, omega_x, omega_y, modulation_kind, omega_0=np.nan, epsilon=0.0, offset=0, force_CPU=False):
        # check if system has a nvidia gpu available
        GPU = gpu_available()
        if force_CPU or not GPU:
            self.tracker = cpp_henon_tracker(N, omega_x, omega_y, modulation_kind, omega_0, epsilon, offset)
        else:
            print("creating gpu tracker")
            self.tracker = cpp_henon_tracker_gpu(
                N, omega_x, omega_y, modulation_kind, omega_0, epsilon, offset)

    def compute_a_modulation(self, n_turns, omega_x, omega_y, epsilon=0.0, modulation_kind="sps", omega_0=np.nan, offset=0):
        self.tracker.compute_a_modulation(n_turns, omega_x, omega_y, modulation_kind, omega_0, epsilon, offset)

    def track(self, particles, n_turns, mu, barrier=10.0, kick_module=np.nan, inverse=False):
        self.tracker.track(particles.particles, n_turns, mu, barrier, kick_module, inverse)

    def tune_birkhoff(self, particles, n_turns, mu, barrier=10.0, kick_module=np.nan, inverse=False, from_idx=np.array([]), to_idx=np.array([])):
        result = np.asarray(self.tracker.birkhoff_tunes(particles.particles, n_turns, mu, barrier, kick_module, inverse, from_idx, to_idx))
        pd_result = pd.DataFrame(columns=["from", "to", "tune_x", "tune_y"])
        for i in range(len(from_idx)):
            pd_result.loc[i] = [from_idx[i], to_idx[i], result[:, i*2], result[:, i*2+1]]
        # add a row to the dataframe
        pd_result.loc[len(from_idx)] = [0, n_turns, result[:, -2], result[:, -1]]
        return pd_result 


class storage():
    def __init__(self, N):
        self.storage = storage_4d(N)

    def store(self, particles):
        self.storage.store(particles.particles)
    
    def tune_fft(self, from_idx, to_idx, max_value):
        result = np.asarray(self.storage.tune_fft(from_idx, to_idx))
        pd_result = pd.DataFrame(columns=["from", "to", "tune_x", "tune_y"])
        for i in range(len(from_idx)):
            pd_result.loc[i] = [from_idx[i], to_idx[i], result[:, i*2], result[:, i*2+1]]
        # add a row to the dataframe
        pd_result.loc[len(from_idx)] = [0, max_value, result[:, -2], result[:, -1]]
        return pd_result

    def tune_birkhoff(self, from_idx, to_idx, max_value):
        result = np.asarray(self.storage.tune_birkhoff(from_idx, to_idx))
        pd_result = pd.DataFrame(columns=["from", "to", "tune_x", "tune_y"])
        for i in range(len(from_idx)):
            pd_result.loc[i] = [from_idx[i], to_idx[i], result[:, i*2], result[:, i*2+1]]
        # add a row to the dataframe
        pd_result.loc[len(from_idx)] = [0, max_value, result[:, -2], result[:, -1]]
        return pd_result 

    def get_x(self):
        return np.asarray(self.storage.get_x())

    def get_px(self):
        return np.asarray(self.storage.get_px())

    def get_y(self):
        return np.asarray(self.storage.get_y())

    def get_py(self):
        return np.asarray(self.storage.get_py())
