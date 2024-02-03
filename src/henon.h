#ifndef HENON_CPU_
#define HENON_CPU_

#ifdef PYBIND
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#endif
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>

#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <vector>
// include standard thread library
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
// include std function
#include <algorithm>
#include <functional>

#include "dynamic_indicator.h"
#include "modulation.h"
#include "utils.h"

#ifndef NT
#define NT 256
#endif // NT

#if defined(USE_CUDA)
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

#if defined(USE_CUDA)
#define CUDA_HOST __host__
#else
#define CUDA_HOST
#endif

struct particles_4d {
  std::vector<double> x;
  std::vector<double> px;
  std::vector<double> y;
  std::vector<double> py;

  std::vector<double> x0;
  std::vector<double> px0;
  std::vector<double> y0;
  std::vector<double> py0;

  std::vector<int> steps;
  std::vector<uint8_t> valid;
  std::vector<uint8_t> ghost;

  size_t n_particles;
  size_t n_ghosts_per_particle;

  std::vector<size_t> idx;
  std::vector<size_t> idx_base;

  std::mt19937_64 rng;
  int global_steps;

  particles_4d() = default;

  particles_4d(const std::vector<double> &x_, const std::vector<double> &px_,
               const std::vector<double> &y_, const std::vector<double> &py_);

  virtual void reset();
  virtual void add_ghost(const double &displacement_module,
                         const std::string &direction);

  virtual void renormalize(const double &module_target);

  virtual const std::vector<std::vector<double>>
  get_displacement_module() const;
  virtual const std::vector<std::vector<std::vector<double>>>
  get_displacement_direction() const;

  virtual const std::vector<double> get_x() const;
  virtual const std::vector<double> get_px() const;
  virtual const std::vector<double> get_y() const;
  virtual const std::vector<double> get_py() const;
  virtual const std::vector<int> get_steps() const;

  virtual std::vector<double> get_radius() const;
  virtual double get_radius_mean() const;
  virtual double get_radius_std() const;
  virtual std::vector<double> get_action() const;
  virtual double get_action_mean() const;
  virtual double get_action_std() const;
  virtual std::vector<double> get_action_x() const;
  virtual double get_action_x_mean() const;
  virtual double get_action_x_std() const;
  virtual std::vector<double> get_action_y() const;
  virtual double get_action_y_mean() const;
  virtual double get_action_y_std() const;
  virtual std::vector<double> get_angles_x() const;
  virtual double get_angles_x_mean() const;
  virtual double get_angles_x_std() const;
  virtual std::vector<double> get_angles_y() const;
  virtual double get_angles_y_mean() const;
  virtual double get_angles_y_std() const;

  virtual const std::vector<uint8_t> get_valid() const;
  virtual const std::vector<uint8_t> get_ghost() const;

  virtual const std::vector<size_t> get_idx() const;
  virtual const std::vector<size_t> get_idx_base() const;

  virtual const size_t &get_n_particles() const;
  virtual const size_t &get_n_ghosts_per_particle() const;
};

class henon_tracker {
protected:
  size_t allowed_steps;

  std::vector<double> omega_x_vec;
  std::vector<double> omega_y_vec;

  std::vector<double> omega_x_sin;
  std::vector<double> omega_x_cos;
  std::vector<double> omega_y_sin;
  std::vector<double> omega_y_cos;

public:
  henon_tracker() = default;
  henon_tracker(unsigned int N, double omega_x, double omega_y,
                std::string modulation_kind = "sps", double omega_0 = NAN,
                double epsilon = 0.0, int offset = 0);

  virtual void compute_a_modulation(unsigned int N, double omega_x,
                                    double omega_y,
                                    std::string modulation_kind = "sps",
                                    double omega_0 = NAN, double epsilon = 0.0,
                                    int offset = 0);

  void track(particles_4d &particles, unsigned int n_turns, double mu,
             double barrier = 100.0, double kick_module = NAN,
             bool inverse = false);

  std::vector<std::vector<double>>
  megno(particles_4d &particles, unsigned int n_turns, double mu,
        double barrier = 100.0, double kick_module = NAN, bool inverse = false,
        std::vector<unsigned int> turn_samples = std::vector<unsigned int>(),
        int n_threads = -1);

  std::vector<std::vector<double>> birkhoff_tunes(
      particles_4d &particles, unsigned int n_turns, double mu,
      double barrier = 100.0, double kick_module = NAN, bool inverse = false,
      std::vector<unsigned int> from_idx = std::vector<unsigned int>(),
      std::vector<unsigned int> to_idx = std::vector<unsigned int>());
  std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
  all_tunes(particles_4d &particles, unsigned int n_turns, double mu,
            double barrier = 100.0, double kick_module = NAN,
            bool inverse = false,
            std::vector<unsigned int> from_idx = std::vector<unsigned int>(),
            std::vector<unsigned int> to_idx = std::vector<unsigned int>());

  std::vector<std::vector<std::vector<double>>>
  get_tangent_matrix(const particles_4d &particles, const double &mu,
                     const bool &reverse) const;
};

struct storage_4d {
  std::vector<std::vector<double>> x;
  std::vector<std::vector<double>> px;
  std::vector<std::vector<double>> y;
  std::vector<std::vector<double>> py;

  storage_4d(size_t N);

  void store(const particles_4d &particles);
  //   void store(const particles_4d_gpu &particles);

  std::vector<std::vector<double>>
  tune_fft(std::vector<unsigned int> from_idx,
           std::vector<unsigned int> to_idx) const;
  std::vector<std::vector<double>>
  tune_birkhoff(std::vector<unsigned int> from_idx,
                std::vector<unsigned int> to_idx) const;

  const std::vector<std::vector<double>> &get_x() const;
  const std::vector<std::vector<double>> &get_px() const;
  const std::vector<std::vector<double>> &get_y() const;
  const std::vector<std::vector<double>> &get_py() const;
};

struct matrix_4d_vector {
  std::vector<std::vector<std::vector<double>>> matrix;

  matrix_4d_vector(size_t N);

  void reset();
  void multiply(const std::vector<std::vector<std::vector<double>>> &l_matrix);
  void structured_multiply(const henon_tracker &tracker,
                           const particles_4d &particles, const double &mu,
                           const bool &reverse);
  // void structured_multiply(const henon_tracker_gpu &tracker, const
  // particles_4d_gpu &particles, const double &mu, const bool &reverse);

  const std::vector<std::vector<std::vector<double>>> &get_matrix() const;
  std::vector<std::vector<double>>
  get_vector(const std::vector<std::vector<double>> &rv) const;
};

// CPU/GPU functions

CUDA_HOST_DEVICE bool check_barrier(const double &x, const double &px,
                                    const double &y, const double &py,
                                    const double &barrier_pow_2);

CUDA_HOST_DEVICE bool check_nan(const double &x, const double &px,
                                const double &y, const double &py);

CUDA_HOST_DEVICE void set_to_nan(double &x, double &px, double &y, double &py);

CUDA_HOST_DEVICE void sextupole(const double &x, double &px, const double &y,
                                double &py, const bool &reverse);

CUDA_HOST_DEVICE void octupole(const double &x, double &px, const double &y,
                               double &py, const double &mu,
                               const bool &reverse);

CUDA_HOST_DEVICE void rotation(double &x, double &px, const double &sin,
                               const double &cos, const bool &reverse);

CUDA_HOST_DEVICE double displacement(const double &x1, const double &px1,
                                     const double &y1, const double &py1,
                                     const double &x2, const double &px2,
                                     const double &y2, const double &py2);

CUDA_HOST_DEVICE void realign(double &x, double &px, double &y, double &py,
                              const double &x_center, const double &px_center,
                              const double &y_center, const double &py_center,
                              const double &initial_module,
                              const double &final_module);

CUDA_HOST std::vector<std::vector<double>>
tangent_matrix(const double &x, const double &px, const double &y,
               const double &py, const double &sx, const double &cx,
               const double &sy, const double &cy, const double &mu);

CUDA_HOST std::vector<std::vector<double>>
inverse_tangent_matrix(const double &x, const double &px, const double &y,
                       const double &py, const double &sx, const double &cx,
                       const double &sy, const double &cy, const double &mu);

#endif // HENON_CPU_