#ifndef DYNAMIC_INDICATOR_H
#define DYNAMIC_INDICATOR_H

#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <complex>
#include <numeric>
#include <array>
#include <map>
#include <thread>
// include fftw libraries
#include <fftw3.h>
#include <mutex>

std::vector<double> birkhoff_weights(unsigned int n_weights);

double birkhoff_tune(std::vector<double> const &x, std::vector<double> const &px);

std::vector<double> birkhoff_tune_vec(std::vector<double> const &x, std::vector<double> const &px, std::vector<double> const &y, std::vector<double> const &py, std::vector<unsigned int> const &from = std::vector<unsigned int>(), std::vector<unsigned int> const &to = std::vector<unsigned int>());

std::map<unsigned int, std::tuple<fftw_complex *, fftw_complex *, fftw_plan>> fft_allocate(std::vector<unsigned int> const &n_points);

void fft_free(std::map<unsigned int, std::tuple<fftw_complex *, fftw_complex *, fftw_plan>> &fft_map);

double fft_tune(
    std::vector<double> const &x, std::vector<double> const &px,
    fftw_complex *in, fftw_complex *out, fftw_plan plan);

std::vector<double> fft_tune_vec(
    std::vector<double> const &x, std::vector<double> const &px,
    std::vector<double> const &y, std::vector<double> const &py,
    std::vector<unsigned int> const &from,
    std::vector<unsigned int> const &to,
    std::map<unsigned int, std::tuple<fftw_complex *, fftw_complex *, fftw_plan>> plans);

std::array<double, 2> get_tunes(std::vector<double> x, std::vector<double> px);

#endif // DYNAMIC_INDICATOR_H