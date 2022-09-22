#include <iostream>
#include <vector>
#include <array>
#include "henon.h"

int main()
{
    std::vector<double> x = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // multiply x by 0.01
    for (auto i = 0; i < x.size(); i++)
    {
        x[i] *= 0.01;
    }

    particles_4d p(x, x, x, x);
    henon_tracker t(1000, 0.31, 0.32);
    
    auto val = t.all_tunes(p, 1000, 0.0);
}