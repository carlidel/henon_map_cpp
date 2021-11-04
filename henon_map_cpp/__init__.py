import numpy as np
from numba import njit, prange
import numba.cuda

from .henon_map_engine import cpu_henon, gpu_henon

class henon_tracker():
    def __init__(self, x_0, px_0, y_0, py_0, omega_x, omega_y, force_CPU=False):
        # check if the first 5 arguments are numpy arrays of the same 1D shape
        if not (isinstance(x_0, np.ndarray) and isinstance(px_0, np.ndarray) and
                isinstance(y_0, np.ndarray) and isinstance(py_0, np.ndarray)):
            raise TypeError("Arguments must be numpy arrays.")
        if not (x_0.ndim == 1 and px_0.ndim == 1 and y_0.ndim == 1 and
                py_0.ndim == 1):
            raise TypeError("Arguments must be 1D arrays.")
        if not (x_0.shape == px_0.shape == y_0.shape == py_0.shape):
            raise ValueError("Arguments must be of the same shape.")     
        # check if the last 2 arguments are numbers
        if not (isinstance(omega_x, (int, float)) and isinstance(omega_y, (int, float))):
            raise TypeError("Arguments must be numbers.")
        # check if the optional argument is a boolean
        if not (isinstance(force_CPU, bool)):
            raise TypeError("Argument must be a boolean.")
        # check if system has a nvidia gpu available with numba
        GPU = numba.cuda.is_available()
        if force_CPU or not GPU:
            self.engine = cpu_henon(x_0, px_0, y_0, py_0, omega_x, omega_y)
        else:
            self.engine = gpu_henon(x_0, px_0, y_0, py_0, omega_x, omega_y)

    def reset(self):
        """Reset the tracker to the initial conditions.
        """
        self.engine.reset()

    def track(self, n_turns, epsilon, mu, barrier=10.0, kick_module=np.nan,
              kick_sigma=np.nan, inverse=False, modulation_kind="sps",
              omega_0=np.nan):
        """Track the system for n_turns turns.

        Parameters
        ----------
        n_turns : unsigned int
            number of turns to track
        epsilon : float
            intensity of the modulation
        mu : float
            intensity of the octupolar kick
        barrier : float, optional
            radial distance of the barrier, by default 10
        kick_module : float, optional
            if desired, average of the kick for every turn, by default np.nan
        kick_sigma : float, optional
            if desired, standard deviation of the kick for every turn, by default np.nan
        inverse : bool, optional
            perform an inverse tracking if True, by default False
        modulation_kind : str, optional
            kind of modulation, either "sps" or "basic", by default "sps"
        omega_0 : float, optional
            modulation harmonic for "basic" option, by default np.nan
        """          
        # check if the arguments are of correct type
        if not (isinstance(n_turns, int) and isinstance(epsilon, (int, float)) and
                isinstance(mu, (int, float)) and isinstance(barrier, (int, float)) and
                isinstance(kick_module, (int, float)) and isinstance(kick_sigma, (int, float)) and
                isinstance(inverse, bool) and isinstance(modulation_kind, str) and
                isinstance(omega_0, (int, float))):
            raise TypeError("Arguments must be of correct type.")
        self.engine.track(n_turns, epsilon, mu, barrier*barrier, kick_module,
            kick_sigma, inverse, modulation_kind, omega_0)

    def get_x(self):
        return np.asarray(self.engine.get_x())

    def get_px(self):
        return np.asarray(self.engine.get_px())

    def get_y(self):
        return np.asarray(self.engine.get_y())

    def get_py(self):
        return np.asarray(self.engine.get_py())

    def get_steps(self):
        return np.asarray(self.engine.get_steps())
        
    def get_omega_x(self):
        return self.engine.get_omega_x()

    def get_omega_y(self):
        return self.engine.get_omega_y()

    def get_global_steps(self):
        return self.engine.get_global_steps()

    def set_x(self, x):
        # check if the argument is a numpy array of the same 1D shape
        if not (isinstance(x, np.ndarray)):
            raise TypeError("Argument must be a numpy array.")
        if not (x.ndim == 1):
            raise ValueError("Argument must be a 1D array.")
        if not (x.shape == self.engine.get_x().shape):
            raise ValueError("Argument must be of the same shape.")
        self.engine.set_x(x)

    def set_px(self, px):
        # check if the argument is a numpy array of the same 1D shape
        if not (isinstance(px, np.ndarray)):
            raise TypeError("Argument must be a numpy array.")
        if not (px.ndim == 1):
            raise ValueError("Argument must be a 1D array.")
        if not (px.shape == self.engine.get_px().shape):
            raise ValueError("Argument must be of the same shape.")
        self.engine.set_px(px)

    def set_y(self, y):
        # check if the argument is a numpy array of the same 1D shape
        if not (isinstance(y, np.ndarray)):
            raise TypeError("Argument must be a numpy array.")
        if not (y.ndim == 1):
            raise ValueError("Argument must be a 1D array.")
        if not (y.shape == self.engine.get_y().shape):
            raise ValueError("Argument must be of the same shape.")
        self.engine.set_y(y)

    def set_py(self, py):
        # check if the argument is a numpy array of the same 1D shape
        if not (isinstance(py, np.ndarray)):
            raise TypeError("Argument must be a numpy array.")
        if not (py.ndim == 1):
            raise ValueError("Argument must be a 1D array.")
        if not (py.shape == self.engine.get_py().shape):
            raise ValueError("Argument must be of the same shape.")
        self.engine.set_py(py)

    def set_omega_x(self, omega_x):
        # check if the argument is a number
        if not (isinstance(omega_x, (int, float))):
            raise TypeError("Argument must be a number.")
        self.engine.set_omega_x(omega_x)

    def set_omega_y(self, omega_y):
        # check if the argument is a number
        if not (isinstance(omega_y, (int, float))):
            raise TypeError("Argument must be a number.")
        self.engine.set_omega_y(omega_y)

    def set_global_steps(self, global_steps):
        # check if the argument is a number
        if not (isinstance(global_steps, (int, float))):
            raise TypeError("Argument must be a number.")
        self.engine.set_global_steps(global_steps)

    def set_steps(self, steps):
        # check if the argument is a numpy array of the same 1D shape
        if (isinstance(steps, np.ndarray)):
            if not (steps.ndim == 1):
                raise ValueError("Argument must be a 1D array.")
            if not (steps.shape == self.engine.get_steps().shape):
                raise ValueError("Argument must be of the same shape.")
        elif not (isinstance(steps, int)):
            raise TypeError("Argument must be a numpy array or an integer.")
        self.engine.set_steps(steps)


        

        
