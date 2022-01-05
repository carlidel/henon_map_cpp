#ifndef MODULATION_H_
#define MODULATION_H_

#include <cmath>
#include <vector>
#include <array>
#include <assert.h>
#include <thread>
#include <mutex>
#include <random>

std::vector<double> basic_modulation(const double &tune, const double &omega, const double &epsilon, const int &start, const int &end);

std::vector<double> sps_modulation(const double &tune, const double &epsilon, const int &start, const int &end);

std::vector<double> gaussian_modulation(const double &tune, const double &sigma, const int &start, const int &end);

std::vector<double> uniform_modulation(const double &from, const double &to, const int &start, const int &end);

#endif // MODULATION_H