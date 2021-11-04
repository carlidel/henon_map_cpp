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

int main(){
    auto n_samples = 10000;
    std::vector<double> x(n_samples);
    std::vector<double> zeros(n_samples, 0.0);
    // fill the vector with a linear sampling between 0.1 amd 0.2
    for(int i = 0; i < n_samples; i++){
        x[i] = 0.1 + 0.1 / n_samples * i;
    }

    std::cout << "Run the CPU version." << std::endl;

    cpu_henon engine(
        x, zeros, x, zeros,
        0.168, 0.201
    );

    // start a timer
    auto start = std::chrono::high_resolution_clock::now();
    
    engine.track(1'000'000, 64.0, 0.0);

    auto end = std::chrono::high_resolution_clock::now();
    
    // get the vectors
    auto new_x = engine.get_x();
    auto new_y = engine.get_y();
    auto new_px = engine.get_px();
    auto new_py = engine.get_py();
    auto steps = engine.get_steps();

    // print the vector x
    for(int i = 0; i < n_samples; i++){
        std::cout << new_x[i] << " ";
    }
    
    std::cout << std::endl;

    // print the vectors
    for(int i = 0; i < n_samples; i++){
        std::cout << new_x[i] << " " << new_y[i] << " " << new_px[i] << " " << new_py[i] << " " << steps[i] << std::endl;
    }

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

        // start a timer
        auto start_gpu = std::chrono::high_resolution_clock::now();
        
        engine_gpu.track(1'000'000, 64.0, 0.0);

        auto end_gpu = std::chrono::high_resolution_clock::now();
        
        // get the vectors
        auto new_x_gpu = engine_gpu.get_x();
        auto new_y_gpu = engine_gpu.get_y();
        auto new_px_gpu = engine_gpu.get_px();
        auto new_py_gpu = engine_gpu.get_py();
        auto steps_gpu = engine_gpu.get_steps();

        // print the vector x
        for(int i = 0; i < n_samples; i++){
            std::cout << new_x_gpu[i] << " ";
        }
        
        std::cout << std::endl;

        // print the vectors
        for(int i = 0; i < n_samples; i++){
            std::cout << new_x_gpu[i] << " " << new_y_gpu[i] << " " << new_px_gpu[i] << " " << new_py_gpu[i] << " " << steps_gpu[i] << std::endl;
        }

        // print the duration in hours:minutes:seconds:milliseconds
        std::cout << "Time taken by GPU engine: "
              << std::chrono::duration_cast<std::chrono::hours>(end_gpu - start_gpu).count() << ":"
              << std::chrono::duration_cast<std::chrono::minutes>(end_gpu - start_gpu).count() % 60 << ":"
              << std::chrono::duration_cast<std::chrono::seconds>(end_gpu - start_gpu).count() % 60 << ":"
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count() % 1000 << std::endl;
    }
    else {
        std::cout << "No GPU available." << std::endl;
    }

    // print the duration in hours:minutes:seconds:milliseconds
    std::cout << "Time taken by CPU engine: "
              << std::chrono::duration_cast<std::chrono::hours>(end - start).count() << ":"
              << std::chrono::duration_cast<std::chrono::minutes>(end - start).count() % 60 << ":"
              << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() % 60 << ":"
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() % 1000 << std::endl;
    
    


    return 0;
}