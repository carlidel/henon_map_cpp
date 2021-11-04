#ifndef HENON_CU_
#define HENON_CU_

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
// include standard thread library
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include "modulation.h"

#ifndef NT
#define NT 256
#endif // NT

class gpu_henon
{
    double omega_x;
    double omega_y;

    std::vector<double> omega_x_vec;
    std::vector<double> omega_y_vec;

    std::vector<double> omega_x_sin;
    std::vector<double> omega_x_cos;
    std::vector<double> omega_y_sin;
    std::vector<double> omega_y_cos;

    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> px;
    std::vector<double> py;

    std::vector<double> x_0;
    std::vector<double> y_0;
    std::vector<double> px_0;
    std::vector<double> py_0;

    std::vector<unsigned int> steps;
    unsigned int global_steps;

    double *d_x;
    double *d_px;
    double *d_y;
    double *d_py;
    unsigned int *d_steps;

    double *d_omega_x_sin;
    double *d_omega_x_cos;
    double *d_omega_y_sin;
    double *d_omega_y_cos;

    curandState *d_rand_states;

    size_t n_samples;
    size_t n_threads;
    size_t n_blocks;

public:
    gpu_henon(const std::vector<double> &x_init,
              const std::vector<double> &px_init,
              const std::vector<double> &y_init,
              const std::vector<double> &py_init,
              double omega_x,
              double omega_y);
    ~gpu_henon();

    void reset();
    void track(unsigned int n_turns, double epsilon, double mu, double barrier = 100.0, double kick_module = NAN, double kick_sigma = NAN, bool inverse = false, std::string modulation_kind = "sps", double omega_0 = NAN);

    // getters
    std::vector<double> get_x() const;
    std::vector<double> get_px() const;
    std::vector<double> get_y() const;
    std::vector<double> get_py() const;
    std::vector<unsigned int> get_steps() const;
    unsigned int get_global_steps() const;
    double get_omega_x() const;
    double get_omega_y() const;

    // setters
    void set_omega_x(double omega_x);
    void set_omega_y(double omega_y);

    void set_x(std::vector<double> x);
    void set_px(std::vector<double> px);
    void set_y(std::vector<double> y);
    void set_py(std::vector<double> py);
    void set_steps(std::vector<unsigned int> steps);
    void set_steps(unsigned int unique_step);
    void set_global_steps(unsigned int global_steps);
};

class cpu_henon
{
    double omega_x;
    double omega_y;

    std::vector<double> omega_x_vec;
    std::vector<double> omega_y_vec;

    std::vector<double> omega_x_sin;
    std::vector<double> omega_x_cos;
    std::vector<double> omega_y_sin;
    std::vector<double> omega_y_cos;

    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> px;
    std::vector<double> py;

    std::vector<double> x_0;
    std::vector<double> y_0;
    std::vector<double> px_0;
    std::vector<double> py_0;

    std::vector<unsigned int> steps;
    unsigned int global_steps;

    unsigned int n_threads;

    std::mt19937_64 generator;
    std::normal_distribution<double> distribution;

public:
    cpu_henon(const std::vector<double> &x_init,
              const std::vector<double> &px_init,
              const std::vector<double> &y_init,
              const std::vector<double> &py_init,
              double omega_x,
              double omega_y);
    ~cpu_henon();

    void reset();
    void track(unsigned int n_turns, double epsilon, double mu, double barrier = 100.0, double kick_module = NAN, double kick_sigma = NAN, bool inverse = false, std::string modulation_kind = "sps", double omega_0 = NAN);

    // getters
    std::vector<double> get_x() const;
    std::vector<double> get_px() const;
    std::vector<double> get_y() const;
    std::vector<double> get_py() const;
    std::vector<unsigned int> get_steps() const;
    unsigned int get_global_steps() const;
    double get_omega_x() const;
    double get_omega_y() const;

    // setters
    void set_omega_x(double omega_x);
    void set_omega_y(double omega_y);

    void set_x(std::vector<double> x);
    void set_px(std::vector<double> px);
    void set_y(std::vector<double> y);
    void set_py(std::vector<double> py);
    void set_steps(std::vector<unsigned int> steps);
    void set_steps(unsigned int unique_step);
    void set_global_steps(unsigned int global_steps);
};


#endif // HENON_CU_