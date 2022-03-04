#ifndef HENON_CU_
#define HENON_CU_

#ifdef PYBIND
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#endif
#include <iostream>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cmath>
#include <vector>
#include <array>
#include <tuple>
#include <random>
#include <chrono>
#include <string>
#include <limits>
#include <cstdint>
// include standard thread library
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
// include std function
#include <functional>
#include <algorithm>

#include "modulation.h"
#include "dynamic_indicator.h"

#ifndef NT
#define NT 256
#endif // NT


struct particles_4d
{
    std::vector<double> x;
    std::vector<double> px;
    std::vector<double> y;
    std::vector<double> py;

    std::vector<double> x0;
    std::vector<double> px0;
    std::vector<double> y0;
    std::vector<double> py0;

    std::vector<unsigned int> steps;
    std::vector<uint8_t> valid;
    std::vector<uint8_t> ghost;

    size_t n_particles;
    size_t n_ghosts_per_particle;

    std::vector<size_t> idx;
    std::vector<size_t> idx_base;

    std::mt19937_64 rng;
    unsigned int global_steps;

    particles_4d() = default;

    particles_4d(
        const std::vector<double> &x_,
        const std::vector<double> &px_,
        const std::vector<double> &y_,
        const std::vector<double> &py_
    );

    virtual void reset();
    virtual void add_ghost(const double &displacement_module, const std::string &direction);

    virtual void renormalize(const double &module_target);

    virtual const std::vector<std::vector<double>> get_displacement_module() const;
    virtual const std::vector<std::vector<std::vector<double>>> get_displacement_direction() const;

    virtual const std::vector<double> get_x() const;
    virtual const std::vector<double> get_px() const;
    virtual const std::vector<double> get_y() const;
    virtual const std::vector<double> get_py() const;
    virtual const std::vector<unsigned int> get_steps() const;
};


struct particles_4d_gpu : public particles_4d
{
    double *d_x;
    double *d_px;
    double *d_y;
    double *d_py;

    unsigned int *d_steps;
    uint8_t *d_valid;
    uint8_t *d_ghost;

    size_t *d_idx;
    size_t *d_idx_base;

    curandState *d_rng_state;

    void _general_host_to_device_copy();
    void _general_device_to_host_copy();

    particles_4d_gpu() = default;

    particles_4d_gpu(
        const std::vector<double> &x_,
        const std::vector<double> &px_,
        const std::vector<double> &y_,
        const std::vector<double> &py_
    );

    virtual void reset();
    virtual void add_ghost(const double &displacement_module, const std::string &direction);

    virtual void renormalize(const double &module_target);

    virtual const std::vector<std::vector<double>> get_displacement_module() const;
    virtual const std::vector<std::vector<std::vector<double>>> get_displacement_direction() const;

    virtual const std::vector<double> get_x() const;
    virtual const std::vector<double> get_px() const;
    virtual const std::vector<double> get_y() const;
    virtual const std::vector<double> get_py() const;
    virtual const std::vector<unsigned int> get_steps() const;

    virtual ~particles_4d_gpu();
    size_t _optimal_nblocks() const;
};


class henon_tracker
{
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
    henon_tracker(unsigned int N, double omega_x, double omega_y, std::string modulation_kind = "sps", double omega_0 = NAN, double epsilon = 0.0, unsigned int offset = 0);

    virtual void compute_a_modulation(unsigned int N, double omega_x, double omega_y, std::string modulation_kind = "sps", double omega_0 = NAN, double epsilon = 0.0, unsigned int offset = 0);

    void track(particles_4d &particles, unsigned int n_turns, double mu, double barrier = 100.0, double kick_module = NAN, bool inverse = false);
};

class henon_tracker_gpu : public henon_tracker
{
protected:
    double *d_omega_x_sin;
    double *d_omega_x_cos;
    double *d_omega_y_sin;
    double *d_omega_y_cos;

    curandState *d_rand_states;
public:
    henon_tracker_gpu() = default;
    henon_tracker_gpu(unsigned int N, double omega_x, double omega_y, std::string modulation_kind = "sps", double omega_0 = NAN, double epsilon = 0.0, unsigned int offset = 0);
    ~henon_tracker_gpu();

    virtual void compute_a_modulation(unsigned int N, double omega_x, double omega_y, std::string modulation_kind = "sps", double omega_0 = NAN, double epsilon = 0.0, unsigned int offset = 0);

    void track(particles_4d_gpu &particles, unsigned int n_turns, double mu, double barrier = 100.0, double kick_module = NAN, bool inverse = false);
};


struct storage_4d
{
    std::vector<std::vector<double>> x;
    std::vector<std::vector<double>> px;
    std::vector<std::vector<double>> y;
    std::vector<std::vector<double>> py;

    storage_4d(size_t N);

    void store(const particles_4d &particles);
    void store(const particles_4d_gpu &particles);

    std::vector<std::vector<double>> tune_fft(std::vector<unsigned int> from_idx, std::vector<unsigned int> to_idx) const;
    std::vector<std::vector<double>> tune_birkhoff(std::vector<unsigned int> from_idx, std::vector<unsigned int> to_idx) const;

    const std::vector<std::vector<double>> &get_x() const;
    const std::vector<std::vector<double>> &get_px() const;
    const std::vector<std::vector<double>> &get_y() const;
    const std::vector<std::vector<double>> &get_py() const;
};

#endif // HENON_CU_