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
    // create vector of linear samples from 0 to 0.1 with 10 samples
    std::vector<double> samples;
    for (int i = 0; i < 10; i++) {
        samples.push_back(i * 0.1);
    }

    particles_4d_gpu particles(samples, samples, samples, samples);
    auto a = particles.get_x();
    particles.add_ghost(1e-10, "random");
    auto b = particles.get_x();
    auto c = particles.get_displacement_module();

    print_vector(a);
    print_vector(b);
    print_vector_vector(c);
}