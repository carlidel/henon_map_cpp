#include <iostream>
#include <vector>
#include <array>
#include "dynamic_indicator.h"

int main()
{
    unsigned int N = 16;
    double tune = 0.21;
    std::vector<double> x(N, 0.0);
    std::vector<double> px(N, 0.0);

    for (unsigned int i = 0; i < N; i++)
    {
        x[i] = sin(2.0 * M_PI * i * tune);
        px[i] = cos(2.0 * M_PI * i * tune);
    }

    auto val = get_tunes(x, px);
}