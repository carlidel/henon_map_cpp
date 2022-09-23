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
#include <set>
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

    virtual const std::vector<uint8_t> get_valid() const;
    virtual const std::vector<uint8_t> get_ghost() const;

    virtual const std::vector<size_t> get_idx() const;
    virtual const std::vector<size_t> get_idx_base() const;

    virtual const size_t &get_n_particles() const;
    virtual const size_t &get_n_ghosts_per_particle() const;
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

    void _general_cudaMalloc();
    void _general_cudaFree();
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

    virtual const std::vector<uint8_t> get_valid() const;
    virtual const std::vector<uint8_t> get_ghost() const;

    virtual const std::vector<size_t> get_idx() const;
    virtual const std::vector<size_t> get_idx_base() const;

    virtual const size_t &get_n_particles() const;
    virtual const size_t &get_n_ghosts_per_particle() const;

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

    std::vector<std::vector<double>> birkhoff_tunes(particles_4d &particles, unsigned int n_turns, double mu, double barrier = 100.0, double kick_module = NAN, bool inverse = false, std::vector<unsigned int> from_idx = std::vector<unsigned int>(), std::vector<unsigned int> to_idx = std::vector<unsigned int>());
    std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>> all_tunes(particles_4d &particles, unsigned int n_turns, double mu, double barrier = 100.0, double kick_module = NAN, bool inverse = false, std::vector<unsigned int> from_idx = std::vector<unsigned int>(), std::vector<unsigned int> to_idx = std::vector<unsigned int>());

    std::vector<std::vector<std::vector<double>>> get_tangent_matrix(const particles_4d &particles, const double &mu, const bool &reverse) const;
};

class henon_tracker_gpu : public henon_tracker
{
public:
    double *d_omega_x_sin;
    double *d_omega_x_cos;
    double *d_omega_y_sin;
    double *d_omega_y_cos;

    curandState *d_rand_states;

    henon_tracker_gpu() = default;
    henon_tracker_gpu(unsigned int N, double omega_x, double omega_y, std::string modulation_kind = "sps", double omega_0 = NAN, double epsilon = 0.0, unsigned int offset = 0);
    ~henon_tracker_gpu();

    virtual void compute_a_modulation(unsigned int N, double omega_x, double omega_y, std::string modulation_kind = "sps", double omega_0 = NAN, double epsilon = 0.0, unsigned int offset = 0);

    void track(particles_4d_gpu &particles, unsigned int n_turns, double mu, double barrier = 100.0, double kick_module = NAN, bool inverse = false);

    std::vector<std::vector<std::vector<double>>> get_tangent_matrix(const particles_4d_gpu &particles, const double &mu, const bool &reverse) const;
};

struct matrix_4d_vector
{
    std::vector<std::vector<std::vector<double>>> matrix;

    matrix_4d_vector(size_t N);

    void reset();
    void multiply(const std::vector<std::vector<std::vector<double>>> &l_matrix);
    void structured_multiply(const henon_tracker &tracker, const particles_4d &particles, const double &mu, const bool &reverse);
    void structured_multiply(const henon_tracker_gpu &tracker, const particles_4d_gpu &particles, const double &mu, const bool &reverse);

    const std::vector<std::vector<std::vector<double>>> &get_matrix() const;
    std::vector<std::vector<double>> get_vector(const std::vector<std::vector<double>> &rv) const;
};

struct matrix_4d_vector_gpu
{
    double *d_matrix;
    size_t N;
    size_t n_blocks;

    matrix_4d_vector_gpu(size_t _N);
    ~matrix_4d_vector_gpu();

    void reset();
    void structured_multiply(const henon_tracker_gpu &tracker, const particles_4d_gpu &particles, const double &mu);
    void set_with_tracker(const henon_tracker_gpu &tracker, const particles_4d_gpu &particles, const double &mu);

    const std::vector<std::vector<std::vector<double>>> get_matrix() const;
    std::vector<std::vector<double>> get_vector(const std::vector<std::vector<double>> &rv) const;
};

struct vector_4d_gpu
{
    double *d_vectors;
    size_t N;
    size_t n_blocks;

    vector_4d_gpu(const std::vector<std::vector<double>> &rv);
    ~vector_4d_gpu();

    void set_vectors(const std::vector<std::vector<double>> &rv);
    void set_vectors(const std::vector<double> &rv);
    void multiply(const matrix_4d_vector_gpu &matrix);
    void normalize();

    const std::vector<std::vector<double>> get_vectors() const;
};

struct lyapunov_birkhoff_construct
{
    double *d_vector;
    double *d_vector_b;
    double *d_birkhoff;
    size_t N;
    size_t n_blocks;
    size_t n_weights;
    size_t idx;

    lyapunov_birkhoff_construct(size_t _N, size_t _n_weights);
    ~lyapunov_birkhoff_construct();

    void reset();
    void change_weights(size_t _n_weights);
    void add(const vector_4d_gpu &vectors);

    std::vector<double> get_weights() const;
    std::vector<double> get_values_raw() const;
    std::vector<double> get_values_b() const;
};

struct lyapunov_birkhoff_construct_multi
{
    std::vector<double*> d_vector;
    std::vector<double*> d_vector_b;
    std::vector<double*> d_birkhoff;
    size_t N;
    size_t n_blocks;
    std::vector<size_t> n_weights;
    size_t idx;

    lyapunov_birkhoff_construct_multi(size_t _N, std::vector<size_t> _n_weights);
    ~lyapunov_birkhoff_construct_multi();

    void reset();
    void add(const vector_4d_gpu &vectors);

    std::vector<std::vector<double>> get_weights() const;
    std::vector<std::vector<double>> get_values_raw() const;
    std::vector<std::vector<double>> get_values_b() const;
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

struct storage_4d_gpu
{
    size_t N;
    size_t batch_size;
    size_t n_blocks;
    size_t idx;
    double *d_x;
    double *d_px;
    double *d_y;
    double *d_py;

    storage_4d_gpu(size_t _N, size_t _batch_size);
    ~storage_4d_gpu();

    void store(const particles_4d_gpu &particles);
    void reset();

    const std::vector<std::vector<double>> get_x() const;
    const std::vector<std::vector<double>> get_px() const;
    const std::vector<std::vector<double>> get_y() const;
    const std::vector<std::vector<double>> get_py() const;
};

#endif // HENON_CU_