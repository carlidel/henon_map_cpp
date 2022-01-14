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
// include standard thread library
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
// include std function
#include <functional>

#include "modulation.h"
#include "dynamic_indicator.h"

#ifndef NT
#define NT 256
#endif // NT

class henon
{
protected:

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

    size_t n_threads_cpu;

    std::mt19937_64 generator;
    std::normal_distribution<double> distribution;

    void compute_a_modulation(unsigned int N, bool inverse=false, std::string modulation_kind="sps", double omega_0=NAN, double epsilon=0.0);

public:
    // default constructor
    henon() = default;

    henon(const std::vector<double> &x_init,
          const std::vector<double> &px_init,
          const std::vector<double> &y_init,
          const std::vector<double> &py_init,
          double omega_x,
          double omega_y);

    virtual void reset();
    virtual void track(unsigned int n_turns, double epsilon, double mu, double barrier = 100.0, double kick_module = NAN, double kick_sigma = NAN, bool inverse = false, std::string modulation_kind = "sps", double omega_0 = NAN) = 0;

    std::array<std::vector<std::vector<double>>, 4> full_track(unsigned int n_turns, double epsilon, double mu, double barrier = 100.0, double kick_module = NAN, double kick_sigma = NAN, std::string modulation_kind = "sps", double omega_0 = NAN);

    std::vector<std::vector<double>> full_track_and_lambda(unsigned int n_turns, double epsilon, double mu, double barrier = 100.0, double kick_module = NAN, double kick_sigma = NAN, std::string modulation_kind = "sps", double omega_0 = NAN, std::function<std::vector<double>(std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>) > lambda = [](std::vector<double> x, std::vector<double> px, std::vector<double> y, std::vector<double> py) {return std::vector<double>{0.0};});

    std::vector<std::vector<double>> birkhoff_tunes(unsigned int n_turns, double epsilon, double mu, double barrier = 100.0, double kick_module = NAN, double kick_sigma = NAN, std::string modulation_kind = "sps", double omega_0 = NAN, std::vector<unsigned int> from=std::vector<unsigned int>(), std::vector<unsigned int> to=std::vector<unsigned int>());

    std::vector<std::vector<double>> fft_tunes(unsigned int n_turns, double epsilon, double mu, double barrier = 100.0, double kick_module = NAN, double kick_sigma = NAN, std::string modulation_kind = "sps", double omega_0 = NAN, std::vector<unsigned int> from=std::vector<unsigned int>(), std::vector<unsigned int> to=std::vector<unsigned int>());

    // getters
    std::vector<double> get_x() const;
    std::vector<double> get_px() const;
    std::vector<double> get_y() const;
    std::vector<double> get_py() const;

    std::vector<double> get_x0() const;
    std::vector<double> get_px0() const;
    std::vector<double> get_y0() const;
    std::vector<double> get_py0() const;

    std::vector<unsigned int> get_steps() const;
    unsigned int get_global_steps() const;
    double get_omega_x() const;
    double get_omega_y() const;

    // setters
    void set_omega_x(double omega_x);
    void set_omega_y(double omega_y);

    virtual void set_x(std::vector<double> x);
    virtual void set_px(std::vector<double> px);
    virtual void set_y(std::vector<double> y);
    virtual void set_py(std::vector<double> py);
    virtual void set_steps(std::vector<unsigned int> steps);
    virtual void set_steps(unsigned int unique_step);
    void set_global_steps(unsigned int global_steps);
};


class gpu_henon : public henon
{
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

    void reset() override;
    void track(unsigned int n_turns, double epsilon, double mu, double barrier = 100.0, double kick_module = NAN, double kick_sigma = NAN, bool inverse = false, std::string modulation_kind = "sps", double omega_0 = NAN) override;

    std::vector<std::vector<double>> track_MEGNO(std::vector<unsigned int> n_turns, double epsilon, double mu, double barrier = 100.0, double kick_module = NAN, double kick_sigma = NAN, bool inverse = false, std::string modulation_kind = "sps", double omega_0 = NAN);

    // Setters
    void set_x(std::vector<double> x) override;
    void set_px(std::vector<double> px) override;
    void set_y(std::vector<double> y) override;
    void set_py(std::vector<double> py) override;
    void set_steps(std::vector<unsigned int> steps) override;
    void set_steps(unsigned int unique_step) override;
};

class cpu_henon : public henon
{
public:
    cpu_henon(const std::vector<double> &x_init,
              const std::vector<double> &px_init,
              const std::vector<double> &y_init,
              const std::vector<double> &py_init,
              double omega_x,
              double omega_y);
    ~cpu_henon();

    void track(unsigned int n_turns, double epsilon, double mu, double barrier = 100.0, double kick_module = NAN, double kick_sigma = NAN, bool inverse = false, std::string modulation_kind = "sps", double omega_0 = NAN) override;
};


#endif // HENON_CU_