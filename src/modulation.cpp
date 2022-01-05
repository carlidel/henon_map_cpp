#include "modulation.h"

std::vector<double> basic_modulation(const double &tune, const double &omega, const double &epsilon, const int &start, const int &end)
{
    assert(start < end);
    std::vector<double> modulation(end - start);
    for (int i = 0; i < end - start; i++)
    {
        modulation[i] = (2 * M_PI * tune * (1 + epsilon * std::cos(2 * M_PI * omega * i)));
    }
    return modulation;
}

std::array<double, 7> sps_coefficients({1.000e-4,
                                        0.218e-4,
                                        0.708e-4,
                                        0.254e-4,
                                        0.100e-4,
                                        0.078e-4,
                                        0.218e-4});

std::array<double, 7> sps_modulations({1 * (2 * M_PI / 868.12),
                                       2 * (2 * M_PI / 868.12),
                                       3 * (2 * M_PI / 868.12),
                                       6 * (2 * M_PI / 868.12),
                                       7 * (2 * M_PI / 868.12),
                                       10 * (2 * M_PI / 868.12),
                                       12 * (2 * M_PI / 868.12)});

std::vector<double> sps_modulation(const double &tune, const double &epsilon, const int &start, const int &end)
{
    assert(start < end);
    std::vector<double> modulation(end - start);
    for (int i = 0; i < end - start; i++)
    {
        auto sum = 0.0;
        for (int j = 0; j < sps_modulations.size(); j++)
        {
            sum += sps_coefficients[j] * std::cos(sps_modulations[j] * i);
        }
        modulation[i] = (2 * M_PI * tune * (1 + epsilon * sum));
    }
    return modulation;
}

std::vector<double> gaussian_modulation(const double &tune, const double &sigma, const int &start, const int &end)
{
    assert(start < end);
    // create random engine
    std::random_device rd;
    std::mt19937 gen(rd());
    // create normal distribution
    std::normal_distribution<double> dist(tune, sigma);

    std::vector<double> modulation(end - start);
    for (int i = 0; i < end - start; i++)
    {
        modulation[i] = (2 * M_PI * dist(gen));
    }

    return modulation;
}

std::vector<double> uniform_modulation(const double &from, const double &to, const int &start, const int &end)
{
    assert(start < end);
    // create random engine
    std::random_device rd;
    std::mt19937 gen(rd());
    // create uniform distribution
    std::uniform_real_distribution<double> dist(from, to);

    std::vector<double> modulation(end - start);
    for (int i = 0; i < end - start; i++)
    {
        modulation[i] = (2 * M_PI * dist(gen));
    }

    return modulation;
}