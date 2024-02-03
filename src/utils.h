#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <cmath>
#include <limits>

#ifdef PYBIND
#include <pybind11/pybind11.h>
#endif // PYBIND

void check_keyboard_interrupt();

double nan_mean(const std::vector<double> &vec);

double nan_std(const std::vector<double> &vec);

#ifdef USE_CUDA
#include <cuda_runtime.h>
void check_cuda_errors();
#endif // USE_CUDA

#endif // UTILS_H