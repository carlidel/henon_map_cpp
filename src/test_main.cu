#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <random>
#include <chrono>
#include <cuda_runtime.h>

#include "henon.h"

// printer function for a vector of vectors
void print_vector_vector(std::vector<std::vector<double>> v) {
    for (int i = 0; i < v.size(); i++) {
        for (int j = 0; j < v[i].size(); j++) {
            std::cout << v[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

// printer function for a vector
template<typename T> void print_vector(std::vector <T> v) {
    for (int i = 0; i < v.size(); i++) {
        std::cout << v[i] << " ";
    }
    std::cout << std::endl;
}


int main(){
    auto n_samples = 10;
    std::vector<double> x(n_samples);
    std::vector<double> zeros(n_samples, 0.0);
    // fill the vector with a linear sampling between 0.1 amd 0.2
    for(int i = 0; i < n_samples; i++){
        x[i] = 0.1 + 0.1 / n_samples * i;
    }

    // std::cout << "Run the CPU version." << std::endl;

    cpu_henon engine(
        x, zeros, x, zeros,
        0.168, 0.201
    );

    engine.track(1'024, 64.0, 0.0);

    // get the new data
    auto x_out = engine.get_x();
    auto y_out = engine.get_y();
    auto px_out = engine.get_px();
    auto py_out = engine.get_py();
    auto steps_out = engine.get_steps();

    // print steps_out
    print_vector(steps_out);    

    auto data_out = engine.full_track(10, 64.0, 0.0);

    auto birkhoff_out = engine.birkhoff_tunes(1024, 64.0, 0.0);
    print_vector_vector(birkhoff_out);

    auto fft_out = engine.fft_tunes(1024, 64.0, 0.0);
    print_vector_vector(fft_out);

    // check if a cuda gpu is available
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    // if a gpu is available, run the gpu version
    if(deviceCount > 0){
        std::cout << "Executing GPU version." << std::endl;

        gpu_henon engine_gpu(
            x, zeros, x, zeros,
            0.168, 0.201
        );

        engine_gpu.track(1'024, 64.0, 0.0);

        // get the vectors
        auto new_x_gpu = engine_gpu.get_x();
        auto new_y_gpu = engine_gpu.get_y();
        auto new_px_gpu = engine_gpu.get_px();
        auto new_py_gpu = engine_gpu.get_py();
        auto steps_gpu = engine_gpu.get_steps();

        // check if cuda errors are present
        cudaError_t error = cudaGetLastError();
        if(error != cudaSuccess){
            std::cout << "Cuda error: " << cudaGetErrorString(error) << std::endl;
        } else {
            std::cout << "No cuda errors." << std::endl;
        }

    }
    else {
        std::cout << "No GPU available." << std::endl;
    }

    return 0;
}