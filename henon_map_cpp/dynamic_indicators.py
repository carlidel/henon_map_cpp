import numpy as np
import pandas as pd
from numba import njit, prange
from tqdm.auto import tqdm

from abc import ABC, abstractmethod
from . import henon_tracker


class abstract_engine(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def track(self, x, px, y, py, t):
        pass

    @abstractmethod
    def keep_tracking(self, t):
        pass

    @abstractmethod
    def track_and_reverse(self, x, px, y, py, t):
        pass


class fixed_henon(abstract_engine):
    def __init__(self, omega_x, omega_y, epsilon=0.0, mu=0.0, barrier=10.0, kick_module=np.nan, kick_sigma=np.nan, modulation_kind="sps", omega_0=np.nan, force_CPU=False):
        self.omega_x = omega_x
        self.omega_y = omega_y
        self.epsilon = epsilon
        self.mu = mu
        self.barrier = barrier
        self.kick_module = kick_module
        self.kick_sigma = kick_sigma
        self.modulation_kind = modulation_kind
        self.omega_0 = omega_0
        self.force_CPU = force_CPU
        
        self.engine = None

    def track(self, x, px, y, py, t):
        self.engine = henon_tracker(x, px, y, py, self.omega_x,
                               self.omega_y, self.force_CPU)

        self.engine.track(t, self.epsilon, self.mu, self.barrier, self.kick_module,
                     self.kick_sigma, False, self.modulation_kind, self.omega_0)

        #print(engine.get_x())
        return self.engine.get_x(), self.engine.get_px(), self.engine.get_y(), self.engine.get_py(), self.engine.get_steps()

    def keep_tracking(self, t):
        assert(self.engine is not None)
        self.engine.track(t, self.epsilon, self.mu, self.barrier, self.kick_module,
                     self.kick_sigma, False, self.modulation_kind, self.omega_0)

        return self.engine.get_x(), self.engine.get_px(), self.engine.get_y(), self.engine.get_py(), self.engine.get_steps()

    def track_and_reverse(self, x, px, y, py, t):
        engine = henon_tracker(x, px, y, py, self.omega_x,
                               self.omega_y, self.force_CPU)

        engine.track(t, self.epsilon, self.mu, self.barrier, self.kick_module,
                     self.kick_sigma, False, self.modulation_kind, self.omega_0)
        engine.track(t, self.epsilon, self.mu, self.barrier, self.kick_module,
                     self.kick_sigma, True, self.modulation_kind, self.omega_0)

        return engine.get_x(), engine.get_px(), engine.get_y(), engine.get_py(), engine.get_steps()


@njit(parallel=True)
def random_4d_displacement(r, module):
    """
    Random 4D displacement
    """
    for i in prange(r.shape[0]):
        r[i] = np.random.uniform(-1.0, 1.0, 4)
        while r[i][0]**2 + r[i][1]**2 > 1.0 or r[i][2]**2 + r[i][3]**2 > 1.0:
            r[i] = np.random.uniform(-1.0, 1.0, 4)
        fix = (1 - r[i][0]**2 - r[i][1]**2) / (r[i][2]**2 + r[i][3]**2)
        r[i][0] *= module
        r[i][1] *= module
        r[i][2] *= fix * module
        r[i][3] *= fix * module
    return r


def fast_lyapunov_indicator(engine, x, px, y, py, t_list, mod_d):
    """
    Fast Lyapunov Indicator
    """
    vec_d = np.empty((x.size, 4))
    vec_d = random_4d_displacement(vec_d, mod_d)
    x_d = x + vec_d[:, 0]
    px_d = px + vec_d[:, 1]
    y_d = y + vec_d[:, 2]
    py_d = py + vec_d[:, 3]

    x_all = np.concatenate((x, x_d))
    px_all = np.concatenate((px, px_d))
    y_all = np.concatenate((y, y_d))
    py_all = np.concatenate((py, py_d))

    t_diff_list = np.concatenate(([t_list[0]], np.diff(t_list)))

    results = pd.DataFrame(columns=['t', 'fast_lyapunov_indicator'])
    # set t column as index
    results.set_index('t', inplace=True)

    for i, t in tqdm(enumerate(t_diff_list), total=len(t_diff_list)):
        if i == 0:
            x_all, px_all, y_all, py_all, _ = engine.track(x_all, px_all, y_all, py_all, t)
        else:
            x_all, px_all, y_all, py_all, _ = engine.keep_tracking(t)
    
        x_out = x_all[:x.size]
        px_out = px_all[:px.size]
        y_out = y_all[:y.size]
        py_out = py_all[:py.size]

        x_d_out = x_all[x.size:]
        px_d_out = px_all[px.size:]
        y_d_out = y_all[y.size:]
        py_d_out = py_all[py.size:]
        
        displacement = np.sqrt((x_d_out - x_out)**2 + (px_d_out - px_out)**2 + (y_d_out - y_out)**2 + (py_d_out - py_out)**2)

        lyap = np.log10(displacement / mod_d) / t_list[i]
        # add lyap to results
        results.loc[t_list[i]] = lyap

    return results


def reversibility_error(engine, x, px, y, py, t_list):
    """
    Reversibility Error
    """
    results = pd.DataFrame(columns=['t', 'reversibility_error'])
    # set t column as index
    results.set_index('t', inplace=True)

    for i, t in tqdm(enumerate(t_list), total=len(t_list)):
        x_rev, px_rev, y_rev, py_rev, _ = engine.track_and_reverse(x, px, y, py, t)

        displacement = np.sqrt((x_rev - x)**2 + (px_rev - px)**2 + (y_rev - y)**2 + (py_rev - py)**2)

        results.loc[t_list[i]] = displacement

    return results


def smallest_alignment_index(engine, x, px, y, py, t_list, tau, mod_d):
    """
    Smallest Alignment Index
    """
    t_max = t_list[-1]
    j = 0
    results = pd.DataFrame(columns=['t', 'smallest_alignment_index'])
    # set t column as index
    results.set_index('t', inplace=True)

    x_d1 = x + mod_d
    px_d1 = px
    y_d1 = y
    py_d1 = py

    x_d2 = x
    px_d2 = px
    y_d2 = y + mod_d
    py_d2 = py

    sali = np.ones_like(x) * np.sqrt(2.0)
    for i in tqdm(range(tau, t_max + tau - 1, tau)):
        if i==tau:
            all_x, all_px, all_y, all_py, _ = engine.track(
                np.concatenate((x, x_d1, x_d2)),
                np.concatenate((px, px_d1, px_d2)),
                np.concatenate((y, y_d1, y_d2)),
                np.concatenate((py, py_d1, py_d2)),
                tau
            )
        else:
            all_x, all_px, all_y, all_py, _ = engine.keep_tracking(tau)

        x = all_x[:x.size]
        px = all_px[:px.size]
        y = all_y[:y.size]
        py = all_py[:py.size]

        x_d1 = all_x[x.size:x.size*2]
        px_d1 = all_px[px.size:px.size*2]
        y_d1 = all_y[y.size:y.size*2]
        py_d1 = all_py[py.size:py.size*2]

        x_d2 = all_x[x.size*2:]
        px_d2 = all_px[px.size*2:]
        y_d2 = all_y[y.size*2:]
        py_d2 = all_py[py.size*2:]

        x_diff_1 = x_d1 - x
        px_diff_1 = px_d1 - px
        y_diff_1 = y_d1 - y
        py_diff_1 = py_d1 - py
        norm_1 = np.sqrt(x_diff_1**2 + px_diff_1**2 + y_diff_1**2 + py_diff_1**2)
        x_diff_1 /= norm_1
        px_diff_1 /= norm_1
        y_diff_1 /= norm_1
        py_diff_1 /= norm_1

        x_diff_2 = x_d2 - x
        px_diff_2 = px_d2 - px
        y_diff_2 = y_d2 - y
        py_diff_2 = py_d2 - py
        norm_2 = np.sqrt(x_diff_2**2 + px_diff_2**2 + y_diff_2**2 + py_diff_2**2)
        x_diff_2 /= norm_2
        px_diff_2 /= norm_2
        y_diff_2 /= norm_2
        py_diff_2 /= norm_2

        sali = np.min([
            sali,
            np.sqrt((x_diff_1 + x_diff_2)**2 + (px_diff_1 + px_diff_2)**2 + (y_diff_1 + y_diff_2)**2 + (py_diff_1 + py_diff_2)**2),
            np.sqrt((x_diff_1 - x_diff_2)**2 + (px_diff_1 - px_diff_2)**2 + (y_diff_1 - y_diff_2)**2 + (py_diff_1 - py_diff_2)**2)
        ], axis=0)

        if i >= t_list[j]:
            results.loc[t_list[j]] = sali
            j += 1

    return results


def global_alignment_index(engine, x, px, y, py, t_list, tau, mod_d):
    """
    Global Alignment Index
    """
    t_max = t_list[-1]
    j = 0
    results = pd.DataFrame(columns=['t', 'global_alignment_index'])
    # set t column as index
    results.set_index('t', inplace=True)

    x_d1 = x + mod_d
    px_d1 = px
    y_d1 = y
    py_d1 = py

    x_d2 = x
    px_d2 = px
    y_d2 = y + mod_d
    py_d2 = py

    x_d3 = x
    px_d3 = px + mod_d
    y_d3 = y
    py_d3 = py

    x_d4 = x
    px_d4 = px
    y_d4 = y
    py_d4 = py + mod_d

    gali = np.ones_like(x)
    for i in tqdm(range(tau, t_max + tau - 1, tau)):
        if i == tau:
            all_x, all_px, all_y, all_py, _ = engine.track(
                np.concatenate((x, x_d1, x_d2, x_d3, x_d4)),
                np.concatenate((px, px_d1, px_d2, px_d3, px_d4)),
                np.concatenate((y, y_d1, y_d2, y_d3, y_d4)),
                np.concatenate((py, py_d1, py_d2, py_d3, py_d4)),
                tau
            )
        else:
            all_x, all_px, all_y, all_py, _ = engine.keep_tracking(tau)

        x = all_x[:x.size]
        px = all_px[:px.size]
        y = all_y[:y.size]
        py = all_py[:py.size]

        x_d1 = all_x[x.size:x.size * 2]
        px_d1 = all_px[px.size:px.size * 2]
        y_d1 = all_y[y.size:y.size * 2]
        py_d1 = all_py[py.size:py.size * 2]

        x_d2 = all_x[x.size * 2: x.size * 3]
        px_d2 = all_px[px.size * 2: px.size * 3]
        y_d2 = all_y[y.size * 2: y.size * 3]
        py_d2 = all_py[py.size * 2: py.size * 3]

        x_d3 = all_x[x.size * 3: x.size * 4]
        px_d3 = all_px[px.size * 3: px.size * 4]
        y_d3 = all_y[y.size * 3: y.size * 4]
        py_d3 = all_py[py.size * 3: py.size * 4]

        x_d4 = all_x[x.size * 4: ]
        px_d4 = all_px[px.size * 4: ]
        y_d4 = all_y[y.size * 4: ]
        py_d4 = all_py[py.size * 4: ]

        x_diff_1 = x_d1 - x
        px_diff_1 = px_d1 - px
        y_diff_1 = y_d1 - y
        py_diff_1 = py_d1 - py
        norm_1 = np.sqrt(x_diff_1**2 + px_diff_1**2 + y_diff_1**2 + py_diff_1**2)
        x_diff_1 /= norm_1
        px_diff_1 /= norm_1
        y_diff_1 /= norm_1
        py_diff_1 /= norm_1

        x_diff_2 = x_d2 - x
        px_diff_2 = px_d2 - px
        y_diff_2 = y_d2 - y
        py_diff_2 = py_d2 - py
        norm_2 = np.sqrt(x_diff_2**2 + px_diff_2**2 + y_diff_2**2 + py_diff_2**2)
        x_diff_2 /= norm_2
        px_diff_2 /= norm_2
        y_diff_2 /= norm_2
        py_diff_2 /= norm_2

        x_diff_3 = x_d3 - x
        px_diff_3 = px_d3 - px
        y_diff_3 = y_d3 - y
        py_diff_3 = py_d3 - py
        norm_3 = np.sqrt(x_diff_3**2 + px_diff_3**2 + y_diff_3**2 + py_diff_3**2)
        x_diff_3 /= norm_3
        px_diff_3 /= norm_3
        y_diff_3 /= norm_3
        py_diff_3 /= norm_3

        x_diff_4 = x_d4 - x
        px_diff_4 = px_d4 - px
        y_diff_4 = y_d4 - y
        py_diff_4 = py_d4 - py
        norm_4 = np.sqrt(x_diff_4**2 + px_diff_4**2 + y_diff_4**2 + py_diff_4**2)
        x_diff_4 /= norm_4
        px_diff_4 /= norm_4
        y_diff_4 /= norm_4
        py_diff_4 /= norm_4

        matrix = np.array([
            [x_diff_1, x_diff_2, x_diff_3, x_diff_4],
            [px_diff_1, px_diff_2, px_diff_3, px_diff_4],
            [y_diff_1, y_diff_2, y_diff_3, y_diff_4],
            [py_diff_1, py_diff_2, py_diff_3, py_diff_4]
        ])
        matrix = np.swapaxes(matrix, 1, 2)
        matrix = np.swapaxes(matrix, 0, 1)

        bool_mask = np.all(np.logical_not(np.isnan(matrix)), axis=(1, 2))
        _, s, _ = np.linalg.svd(matrix[bool_mask], full_matrices=True)
        result = np.zeros((len(x)))
        result[np.logical_not(bool_mask)] = np.nan
        result[bool_mask] = np.prod(s, axis=-1)

        gali = np.min([
            gali,
            result
        ], axis=0)

        if i >= t_list[j]:
            results.loc[t_list[j]] = gali
            j += 1

    return results