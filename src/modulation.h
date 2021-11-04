#ifndef MODULATION_H_
#define MODULATION_H_

#include <cmath>
#include <vector>
#include <array>
#include <assert.h>

std::vector<double> basic_modulation(const double &tune, const double &omega, const double &epsilon, const int &start, const int &end);

std::vector<double> sps_modulation(const double &tune, const double &epsilon, const int &start, const int &end);

#endif // MODULATION_H