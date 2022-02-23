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
    auto n_threads_cpu = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;

    for (unsigned int k = 0; k < n_threads_cpu; k++)
    {
        threads.push_back(std::thread(
            [&](const unsigned int n_th)
            {
                for (unsigned int i = n_th; i < end - start; i+=n_threads_cpu)
                {
                    auto sum = 0.0;
                    for (unsigned int j = 0; j < sps_modulations.size(); j++)
                    {
                        sum += sps_coefficients[j] * std::cos(sps_modulations[j] * (i + start));
                    }
                    modulation[i] = (2 * M_PI * tune * (1 + epsilon * sum));
                }
            },
            k));
    }
    // join threads
    for (auto &t : threads)
    {
        t.join();
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

std::tuple<std::vector<double>, std::vector<double>> pick_a_modulation(unsigned int n_turns, double omega_x, double omega_y, std::string modulation_kind, double omega_0, double epsilon, unsigned int offset)
{
    std::vector<double> omega_x_vec;
    std::vector<double> omega_y_vec;
    // compute a modulation
    if (modulation_kind == "sps")
    {
        omega_x_vec = sps_modulation(omega_x, epsilon, offset, offset + n_turns);
        omega_y_vec = sps_modulation(omega_y, epsilon, offset, offset + n_turns);
    }
    else if (modulation_kind == "basic")
    {
        assert(!std::isnan(omega_0));
        omega_x_vec = basic_modulation(omega_x, omega_0, epsilon, offset, offset + n_turns);
        omega_y_vec = basic_modulation(omega_y, omega_0, epsilon, offset, offset + n_turns);
    }
    else if (modulation_kind == "none")
    {
        std::fill(omega_x_vec.begin(), omega_x_vec.end(), omega_x);
        std::fill(omega_y_vec.begin(), omega_y_vec.end(), omega_y);
    }
    else if (modulation_kind == "gaussian")
    {
        omega_x_vec = gaussian_modulation(omega_x, epsilon, offset, offset + n_turns);
        omega_y_vec = gaussian_modulation(omega_y, epsilon, offset, offset + n_turns);
    }
    else if (modulation_kind == "uniform")
    {
        omega_x_vec = uniform_modulation(omega_x, epsilon, offset, offset + n_turns);
        omega_y_vec = uniform_modulation(omega_y, epsilon, offset, offset + n_turns);
    }
    else
    {
        throw std::runtime_error("Unknown modulation kind");
    }

    return std::make_tuple(omega_x_vec, omega_y_vec);
}