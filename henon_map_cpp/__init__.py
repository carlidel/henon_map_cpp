import numba.cuda
import numpy as np
import pandas as pd

from .henon_map_engine import birkhoff_weights as cpp_birkhoff_weights
from .henon_map_engine import henon_tracker as cpp_henon_tracker
from .henon_map_engine import henon_tracker_gpu as cpp_henon_tracker_gpu
from .henon_map_engine import lyapunov_birkhoff_construct as lbc
from .henon_map_engine import lyapunov_birkhoff_construct_multi as lbcm
from .henon_map_engine import matrix_4d_vector as cpp_matrix_4d_vector
from .henon_map_engine import matrix_4d_vector_gpu as cpp_matrix_4d_vector_gpu
from .henon_map_engine import megno_construct as megno
from .henon_map_engine import megno_birkhoff_construct as megno_birkhoff
from .henon_map_engine import tune_birkhoff_construct as tune_birkhoff
from .henon_map_engine import particles_4d, particles_4d_gpu, storage_4d
from .henon_map_engine import storage_4d_gpu as cpp_storage_4d_gpu
from .henon_map_engine import vector_4d_gpu

# from .henon_map_engine import cpu_henon, gpu_henon, is_cuda_device_available


def gpu_available():
    return numba.cuda.is_available()


def birkhoff_weights(n):
    return np.asarray(cpp_birkhoff_weights(n))


class particles:
    def __init__(self, x_0, px_0, y_0, py_0, force_CPU=False):
        # check if the first 5 arguments are numpy arrays of the same 1D shape
        if not (
            isinstance(x_0, np.ndarray)
            and isinstance(px_0, np.ndarray)
            and isinstance(y_0, np.ndarray)
            and isinstance(py_0, np.ndarray)
        ):
            raise TypeError("Arguments must be numpy arrays.")
        if not (x_0.ndim == 1 and px_0.ndim == 1 and y_0.ndim == 1 and py_0.ndim == 1):
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
        return self.particles.get_x()

    def get_px(self):
        return np.asarray(self.particles.get_px())

    def get_y(self):
        return np.asarray(self.particles.get_y())

    def get_py(self):
        return np.asarray(self.particles.get_py())

    def get_steps(self):
        return np.asarray(self.particles.get_steps())


class matrix_4d_vector:
    def __init__(self, N, force_cpu=False):
        self.GPU = gpu_available()
        if force_cpu or not self.GPU:
            print("Using CPU")
            self.matrix = cpp_matrix_4d_vector(N)
            self.GPU = False
        else:
            print("Using GPU")
            self.matrix = cpp_matrix_4d_vector_gpu(N)

    def reset(self):
        self.matrix.reset()

    def multiply(self, matrices):
        self.matrix.multiply(matrices)

    def structured_multiply(self, tracker, particles, mu, reverse=False):
        if self.GPU:
            self.matrix.structured_multiply(tracker.tracker, particles.particles, mu)
        else:
            self.matrix.structured_multiply(
                tracker.tracker, particles.particles, mu, reverse
            )

    def set_with_tracker(self, tracker, particles, mu, reverse=False):
        if self.GPU:
            self.matrix.set_with_tracker(tracker.tracker, particles.particles, mu)
        else:
            # not implemented...
            raise NotImplementedError

    def explicit_copy(self, other_matrix):
        if self.GPU:
            self.matrix.explicit_copy(other_matrix.matrix)
        else:
            raise NotImplementedError

    def get_matrix(self):
        return np.asarray(self.matrix.get_matrix())

    def get_vector(self, vector):
        return np.asarray(self.matrix.get_vector(vector))


class vector_4d:
    def __init__(self, initial_vector):
        assert gpu_available()
        self.vector = vector_4d_gpu(initial_vector)

    def set_vectors(self, vector):
        self.vector.set_vectors(vector)

    def multiply(self, matrix: matrix_4d_vector):
        self.vector.multiply(matrix.matrix)

    def normalize(self):
        self.vector.normalize()

    def get_vectors(self):
        return np.asarray(self.vector.get_vectors())


class lyapunov_birkhoff_construct:
    def __init__(self, n, n_weights):
        assert gpu_available()
        self.construct = lbc(n, n_weights)

    def reset(self):
        self.construct.reset()

    def change_weights(self, n_weights):
        self.construct.change_weights(n_weights)

    def add(self, vectors):
        self.construct.add(vectors.vector)

    def get_weights(self):
        return np.asarray(self.construct.get_weights())

    def get_values_raw(self):
        return np.asarray(self.construct.get_values_raw())

    def get_values_b(self):
        return np.asarray(self.construct.get_values_b())


class lyapunov_birkhoff_construct_multi:
    def __init__(self, n, n_weights):
        assert gpu_available()
        self.construct = lbcm(n, n_weights)

    def reset(self):
        self.construct.reset()

    def add(self, vectors):
        self.construct.add(vectors.vector)

    def get_values_raw(self):
        return np.asarray(self.construct.get_values_raw())

    def get_values_b(self):
        return np.asarray(self.construct.get_values_b())


class megno_construct:
    def __init__(self, n):
        assert gpu_available()
        self.construct = megno(n)

    def reset(self):
        self.construct.reset()

    def add(self, matrix_a: matrix_4d_vector, matrix_b: matrix_4d_vector):
        self.construct.add(matrix_a.matrix, matrix_b.matrix)

    def get_values(self):
        return np.asarray(self.construct.get_values())


class megno_birkhoff_construct:
    def __init__(self, n, n_weights):
        assert gpu_available()
        self.construct = megno_birkhoff(n, n_weights)

    def reset(self):
        self.construct.reset()

    def add(self, matrix_a: matrix_4d_vector, matrix_b: matrix_4d_vector):
        self.construct.add(matrix_a.matrix, matrix_b.matrix)

    def get_values(self):
        return np.asarray(self.construct.get_values())


class tune_birkhoff_construct:
    def __init__(self, n, n_weights):
        assert gpu_available()
        self.construct = tune_birkhoff(n, n_weights)
        self.first_called = False

    def reset(self):
        self.construct.reset()

    def first_add(self, particles: particles_4d):
        self.construct.first_add(particles.particles)
        self.first_called = True

    def add(self, particles: particles_4d):
        if not self.first_called:
            raise ValueError("First call add_first")
        self.construct.add(particles.particles)
    
    def get_tune1_x(self):
        return self.construct.get_tune1_x()

    def get_tune1_y(self):
        return self.construct.get_tune1_y()
    
    def get_tune2_x(self):
        return self.construct.get_tune2_x()

    def get_tune2_y(self):
        return self.construct.get_tune2_y()


class henon_tracker:
    def __init__(
        self,
        N,
        omega_x,
        omega_y,
        modulation_kind,
        omega_0=np.nan,
        epsilon=0.0,
        offset=0,
        force_CPU=False,
    ):
        # check if system has a nvidia gpu available
        GPU = gpu_available()
        if force_CPU or not GPU:
            self.tracker = cpp_henon_tracker(
                N, omega_x, omega_y, modulation_kind, omega_0, epsilon, offset
            )
        else:
            print("creating gpu tracker")
            self.tracker = cpp_henon_tracker_gpu(
                N, omega_x, omega_y, modulation_kind, omega_0, epsilon, offset
            )

    def compute_a_modulation(
        self,
        n_turns,
        omega_x,
        omega_y,
        epsilon=0.0,
        modulation_kind="sps",
        omega_0=np.nan,
        offset=0,
    ):
        self.tracker.compute_a_modulation(
            n_turns, omega_x, omega_y, modulation_kind, omega_0, epsilon, offset
        )

    def track(
        self, particles, n_turns, mu, barrier=10.0, kick_module=np.nan, inverse=False
    ):
        self.tracker.track(
            particles.particles, n_turns, mu, barrier, kick_module, inverse
        )

    def megno(
        self,
        particles,
        n_turns,
        mu,
        barrier=10.0,
        kick_module=np.nan,
        inverse=False,
        turn_samples=np.array([]),
        n_threads=-1,
    ):
        result = np.asarray(
            self.tracker.megno(
                particles.particles,
                n_turns,
                mu,
                barrier,
                kick_module,
                inverse,
                turn_samples,
                n_threads,
            )
        )
        return result

    def tune_birkhoff(
        self,
        particles,
        n_turns,
        mu,
        barrier=10.0,
        kick_module=np.nan,
        inverse=False,
        from_idx=np.array([]),
        to_idx=np.array([]),
    ):
        result = np.asarray(
            self.tracker.birkhoff_tunes(
                particles.particles,
                n_turns,
                mu,
                barrier,
                kick_module,
                inverse,
                from_idx,
                to_idx,
            )
        )
        pd_result = pd.DataFrame(columns=["from", "to", "tune_x", "tune_y"])
        for i in range(len(from_idx)):
            pd_result.loc[i] = [
                from_idx[i],
                to_idx[i],
                result[:, i * 2],
                result[:, i * 2 + 1],
            ]
        # add a row to the dataframe
        pd_result.loc[len(from_idx)] = [0, n_turns, result[:, -2], result[:, -1]]
        return pd_result

    def tune_all(
        self,
        particles,
        n_turns,
        mu,
        barrier=10.0,
        kick_module=np.nan,
        inverse=False,
        from_idx=np.array([]),
        to_idx=np.array([]),
    ):
        result_birkhoff, result_fft = np.asarray(
            self.tracker.all_tunes(
                particles.particles,
                n_turns,
                mu,
                barrier,
                kick_module,
                inverse,
                from_idx,
                to_idx,
            )
        )
        pd_result = pd.DataFrame(
            columns=[
                "from",
                "to",
                "tune_x_birkhoff",
                "tune_y_birkhoff",
                "tune_x_fft",
                "tune_y_fft",
            ]
        )
        for i in range(len(from_idx)):
            pd_result.loc[i] = [
                from_idx[i],
                to_idx[i],
                result_birkhoff[:, i * 2],
                result_birkhoff[:, i * 2 + 1],
                result_fft[:, i * 2],
                result_fft[:, i * 2 + 1],
            ]
        # add a row to the dataframe
        pd_result.loc[len(from_idx)] = [
            0,
            n_turns,
            result_birkhoff[:, -2],
            result_birkhoff[:, -1],
            result_fft[:, -2],
            result_fft[:, -1],
        ]
        return pd_result

    def get_tangent_matrix(self, particles, mu, reverse=False):
        return self.tracker.get_tangent_matrix(particles.particles, mu, reverse)


class storage:
    def __init__(self, N):
        self.storage = storage_4d(N)

    def store(self, particles):
        self.storage.store(particles.particles)

    def tune_fft(self, from_idx, to_idx, max_value):
        result = np.asarray(self.storage.tune_fft(from_idx, to_idx))
        pd_result = pd.DataFrame(columns=["from", "to", "tune_x", "tune_y"])
        for i in range(len(from_idx)):
            pd_result.loc[i] = [
                from_idx[i],
                to_idx[i],
                result[:, i * 2],
                result[:, i * 2 + 1],
            ]
        # add a row to the dataframe
        pd_result.loc[len(from_idx)] = [0, max_value, result[:, -2], result[:, -1]]
        return pd_result

    def tune_birkhoff(self, from_idx, to_idx, max_value):
        result = np.asarray(self.storage.tune_birkhoff(from_idx, to_idx))
        pd_result = pd.DataFrame(columns=["from", "to", "tune_x", "tune_y"])
        for i in range(len(from_idx)):
            pd_result.loc[i] = [
                from_idx[i],
                to_idx[i],
                result[:, i * 2],
                result[:, i * 2 + 1],
            ]
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


class storage_gpu:
    def __init__(self, N, batch_size):
        self.storage = cpp_storage_4d_gpu(N, batch_size)

    def store(self, particles):
        self.storage.store(particles.particles)

    def reset(self):
        self.storage.reset()

    def get_x(self):
        return np.asarray(self.storage.get_x())

    def get_px(self):
        return np.asarray(self.storage.get_px())

    def get_y(self):
        return np.asarray(self.storage.get_y())

    def get_py(self):
        return np.asarray(self.storage.get_py())
