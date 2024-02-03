#include "utils.h"

#ifdef PYBIND
namespace py = pybind11;
#endif // PYBIND

void check_keyboard_interrupt()
{
#ifdef PYBIND
    if (PyErr_CheckSignals() != 0)
    {
        std::cout << "Keyboard interrupt" << std::endl;
        throw py::error_already_set();
    }
#endif // PYBIND
}

double nan_mean(const std::vector<double> &vec)
{
    double sum = 0.0;
    int count = 0;
    for (auto &v : vec)
    {
        if (!isnan(v))
        {
            sum += v;
            count++;
        }
    }
    if (count == 0)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return sum / count;
}

double nan_std(const std::vector<double> &vec)
{
    double mean = nan_mean(vec);
    double sum = 0.0;
    int count = 0;
    for (auto &v : vec)
    {
        if (!isnan(v))
        {
            sum += (v - mean) * (v - mean);
            count++;
        }
    }
    if (count == 0)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return sqrt(sum / count);
}

#ifdef USE_CUDA

void check_cuda_errors()
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
#ifndef PYBIND
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
#else
        py::print("CUDA error: ", cudaGetErrorString(err));
        throw std::runtime_error("CUDA error");
#endif // PYBIND
    }
    check_keyboard_interrupt();
}

#endif // USE_CUDA