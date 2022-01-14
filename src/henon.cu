#include "henon.h"

#ifdef PYBIND
    namespace py = pybind11;
#endif

std::array<double, 4> random_4d_kick(std::mt19937_64 *generator, std::normal_distribution<double> distribution)
{
    // create random uniform distribution between -1 and 1
    auto uni_dist = std::uniform_real_distribution<double>(-1.0, 1.0);
    // create the array
    std::array<double, 4> kick;
    // fill it with random uniform numbers between -1 and 1
    for (auto& kick_component : kick)
    {
        kick_component = uni_dist(*generator);
    }
    while (kick[0] * kick[0] + kick[1] * kick[1] > 1.0)
    {
        kick[0] = uni_dist(*generator);
        kick[1] = uni_dist(*generator);
    }
    while (kick[2] * kick[2] + kick[3] * kick[3] > 1.0)
    {
        kick[2] = uni_dist(*generator);
        kick[3] = uni_dist(*generator);
    }
    // extract the module from the normal distribution
    auto kick_module = distribution(*generator);
    auto kick_fix = (1 - kick[0] * kick[0] - kick[1] * kick[1]) / (kick[2] * kick[2] + kick[3] * kick[3]);
    // scale the kick
    kick[0] *= kick_module;
    kick[1] *= kick_module;
    kick[2] *= kick_fix * kick_module;
    kick[3] *= kick_fix * kick_module;
    // return the kick
    return kick;
}

void cpu_full_henon_track(
    unsigned int idx,
    double *x,
    double *px,
    double *y,
    double *py,
    const unsigned int max_steps,
    const double barrier,
    const double mu,
    const double *omega_x_sin,
    const double *omega_x_cos,
    const double *omega_y_sin,
    const double *omega_y_cos,
    const bool random_kick,
    std::mt19937_64 *generator,
    std::normal_distribution<double> distribution
)
{
    double temp1;
    double temp2;
    for (unsigned int i = 0; i < max_steps; i++)
    {
        if(isnan(x[i]) || isnan(px[i]) || isnan(y[i]) || isnan(py[i]))
        {
            x[i + 1] = NAN;
            px[i + 1] = NAN;
            y[i + 1] = NAN;
            py[i + 1] = NAN;
            continue;
        }

        temp1 = (px[i] + x[i] * x[i] - y[i] * y[i]);
        temp2 = (py[i] - 2 * x[i] * y[i]);

        if (mu != 0.0)
        {
            temp1 += mu * (x[i] * x[i] * x[i] - 3 * y[i] * y[i] * x[i]);
            temp2 -= mu * (3 * x[i] * x[i] * y[i] - y[i] * y[i] * y[i]);
        }

        px[i + 1] = -omega_x_sin[i] * x[i] + omega_x_cos[i] * temp1;
        x[i + 1] = +omega_x_cos[i] * x[i] + omega_x_sin[i] * temp1;
        py[i + 1] = -omega_y_sin[i] * y[i] + omega_y_cos[i] * temp2;
        y[i + 1] = +omega_y_cos[i] * y[i] + omega_y_sin[i] * temp2;

        if (random_kick)
        {
            auto kick = random_4d_kick(generator, distribution);
            x[i + 1] += kick[0];
            px[i + 1] += kick[1];
            y[i + 1] += kick[2];
            py[i + 1] += kick[3];
        }

        if (x[i + 1] * x[i + 1] + y[i + 1] * y[i + 1] + px[i + 1] * px[i + 1] + py[i + 1] * py[i + 1] > barrier)
        {
            x[i + 1] = NAN;
            px[i + 1] = NAN;
            y[i + 1] = NAN;
            py[i + 1] = NAN;
        }
    }
}


void cpu_henon_track(
    unsigned int idx,
    double *x_g,
    double *px_g,
    double *y_g,
    double *py_g,
    unsigned int *steps_g,
    const unsigned int max_steps,
    const double barrier,
    const double mu,
    const double *omega_x_sin,
    const double *omega_x_cos,
    const double *omega_y_sin,
    const double *omega_y_cos,
    const bool inverse,
    const bool random_kick,
    std::mt19937_64 *generator,
    std::normal_distribution<double> distribution
)
{
    // save local copies of the global variables
    double x = x_g[idx];
    double px = px_g[idx];
    double y = y_g[idx];
    double py = py_g[idx];
    unsigned int steps = steps_g[idx];

    double temp1;
    double temp2;
    if (!inverse)
    {
        for (unsigned int i = 0; i < max_steps; i++)
        {
            if (i == 0)
            {
                if (isnan(x) || isnan(px) || isnan(y) || isnan(py))
                {
                    break;
                }
            }
            temp1 = (px + x * x - y * y);
            temp2 = (py - 2 * x * y);

            if (mu != 0.0)
            {
                temp1 += mu * (x * x * x - 3 * y * y * x);
                temp2 -= mu * (3 * x * x * y - y * y * y);
            }

            px = -omega_x_sin[i] * x + omega_x_cos[i] * temp1;
            x = +omega_x_cos[i] * x + omega_x_sin[i] * temp1;
            py = -omega_y_sin[i] * y + omega_y_cos[i] * temp2;
            y = +omega_y_cos[i] * y + omega_y_sin[i] * temp2;

            if (random_kick)
            {
                auto kick = random_4d_kick(generator, distribution);
                x += kick[0];
                px += kick[1];
                y += kick[2];
                py += kick[3];
            }

            if (x * x + y * y + px * px + py * py > barrier)
            {
                x = NAN;
                px = NAN;
                y = NAN;
                py = NAN;
                break;
            }
            steps += 1;
        }
    }
    else
    {
        for (unsigned int k = max_steps; k != 0; --k)
        {
                if (isnan(x) || isnan(px) || isnan(y) || isnan(py))
                {
                    break;
                }
                temp1 = px;
                temp2 = py;

            px = +omega_x_sin[k - 1] * x + omega_x_cos[k - 1] * temp1;
            x = +omega_x_cos[k - 1] * x - omega_x_sin[k - 1] * temp1;
            py = +omega_y_sin[k - 1] * y + omega_y_cos[k - 1] * temp2;
            y = +omega_y_cos[k - 1] * y - omega_y_sin[k - 1] * temp2;

            px = (px - x * x + y * y);
            py = (py + 2 * x * y);

            if (mu != 0.0)
            {
                px -= mu * (x * x * x - 3 * y * y * x);
                py += mu * (3 * x * x * y - y * y * y);
            }

            if (random_kick)
            {
                auto kick = random_4d_kick(generator, distribution);
                x += kick[0];
                px += kick[1];
                y += kick[2];
                py += kick[3];
            }

            if (x * x + y * y + px * px + py * py > barrier)
            {
                x = NAN;
                px = NAN;
                y = NAN;
                py = NAN;
                break;
            }
            steps -= 1;
        }
    }
    
    // save the results
    x_g[idx] = x;
    px_g[idx] = px;
    y_g[idx] = y;
    py_g[idx] = py;
    steps_g[idx] = steps;
}

__global__ void gpu_henon_track(
    double *g_x,
    double *g_px,
    double *g_y,
    double *g_py,
    unsigned int *g_steps,
    const size_t n_samples,
    const unsigned int max_steps,
    const double barrier,
    const double mu,
    const double *omega_x_sin,
    const double *omega_x_cos,
    const double *omega_y_sin,
    const double *omega_y_cos)
{
    double x;
    double px;
    double y;
    double py;
    double temp1;
    double temp2;
    unsigned int steps;

    size_t j = threadIdx.x + blockIdx.x * blockDim.x;

    if (j < n_samples)
    {
        // Load from shared
        x = g_x[j];
        px = g_px[j];
        y = g_y[j];
        py = g_py[j];
        steps = g_steps[j];

        // Track
        for (unsigned int k = 0; k < max_steps; ++k)
        {
            if (k == 0)
            {
                if (isnan(x) || isnan(px) || isnan(y) || isnan(py))
                {
                    break;
                }
            }

            temp1 = (px + x * x - y * y);
            temp2 = (py - 2 * x * y);
            
            if (mu != 0.0)
            {
                temp1 += mu * (x * x * x - 3 * y * y * x);
                temp2 -= mu * (3 * x * x * y - y * y * y);
            }

            px = -omega_x_sin[k] * x + omega_x_cos[k] * temp1;
            x = +omega_x_cos[k] * x + omega_x_sin[k] * temp1;
            py = -omega_y_sin[k] * y + omega_y_cos[k] * temp2;
            y = +omega_y_cos[k] * y + omega_y_sin[k] * temp2;

            if (x * x + y * y + px * px + py * py > barrier)
            {
                x = NAN;
                px = NAN;
                y = NAN;
                py = NAN;
                break;
            }
            steps += 1;
        }

        // Save in global
        g_x[j] = x;
        g_px[j] = px;
        g_y[j] = y;
        g_py[j] = py;
        g_steps[j] = steps;
    }
}

__global__ void gpu_henon_track_inverse(
    double *g_x,
    double *g_px,
    double *g_y,
    double *g_py,
    unsigned int *g_steps,
    const size_t n_samples,
    const unsigned int max_steps,
    const double barrier,
    const double mu,
    const double *omega_x_sin,
    const double *omega_x_cos,
    const double *omega_y_sin,
    const double *omega_y_cos)
{
    double x;
    double px;
    double y;
    double py;
    double temp1;
    double temp2;
    unsigned int steps;

    size_t j = threadIdx.x + blockIdx.x * blockDim.x;

    if (j < n_samples)
    {
        // Load from shared
        x = g_x[j];
        px = g_px[j];
        y = g_y[j];
        py = g_py[j];
        steps = g_steps[j];

        // Track
        for (unsigned int k = max_steps; k != 0; --k)
        {
                if (isnan(x) || isnan(px) || isnan(y) || isnan(py))
                {
                    break;
                }
                temp1 = px;
                temp2 = py;
            
            px = + omega_x_sin[k - 1] * x + omega_x_cos[k - 1] * temp1;
            x  = + omega_x_cos[k - 1] * x - omega_x_sin[k - 1] * temp1;
            py = + omega_y_sin[k - 1] * y + omega_y_cos[k - 1] * temp2;
            y  = + omega_y_cos[k - 1] * y - omega_y_sin[k - 1] * temp2;

            px = (px - x * x + y * y);
            py = (py + 2 * x * y);
            
            if (mu != 0.0)
            {
                px -= mu * (x * x * x - 3 * y * y * x);
                py += mu * (3 * x * x * y - y * y * y);
            }

            if (x * x + y * y + px * px + py * py > barrier)
            {
                x = NAN;
                px = NAN;
                y = NAN;
                py = NAN;
                break;
            }
            steps -= 1;
        }

        // Save in global
        g_x[j] = x;
        g_px[j] = px;
        g_y[j] = y;
        g_py[j] = py;
        g_steps[j] = steps;
    }
}

__global__ void setup_random_states(curandState *states, unsigned long long seed)
{
	unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, i, 0, &states[i]);
}

__device__ void extract_random_kicks(double* rand_kicks, curandState *state, size_t j, const double kick_sigma, const double kick_average)
{
    rand_kicks[0] = curand_uniform(&state[j]) * 2.0 - 1.0;
    rand_kicks[1] = curand_uniform(&state[j]) * 2.0 - 1.0;
    while (rand_kicks[0] * rand_kicks[0] + rand_kicks[1] * rand_kicks[1] > 1.0) {
        rand_kicks[0] = curand_uniform(&state[j]) * 2.0 - 1.0;
        rand_kicks[1] = curand_uniform(&state[j]) * 2.0 - 1.0;
    }
    rand_kicks[2] = curand_uniform(&state[j]) * 2.0 - 1.0;
    rand_kicks[3] = curand_uniform(&state[j]) * 2.0 - 1.0;
    while (rand_kicks[2] * rand_kicks[2] + rand_kicks[3] * rand_kicks[3] > 1.0) {
        rand_kicks[2] = curand_uniform(&state[j]) * 2.0 - 1.0;
        rand_kicks[3] = curand_uniform(&state[j]) * 2.0 - 1.0;
    }
    double rfix = (1 - rand_kicks[0] * rand_kicks[0] - rand_kicks[1] * rand_kicks[1]) / (rand_kicks[2] * rand_kicks[2] + rand_kicks[3] * rand_kicks[3]);
    // Compute 1 normal random number
    double rm = curand_normal(&state[j]);
    // rescale the normal random number
    rm = rm * kick_sigma + kick_average;
    rand_kicks[0] *= rm;
    rand_kicks[1] *= rm;
    rand_kicks[2] *= rm * rfix;
    rand_kicks[3] *= rm * rfix;
}


__global__ void gpu_henon_track_with_kick(
    double *g_x,
    double *g_px,
    double *g_y,
    double *g_py,
    unsigned int *g_steps,
    const size_t n_samples,
    const unsigned int max_steps,
    const double barrier,
    const double mu,
    const double *omega_x_sin,
    const double *omega_x_cos,
    const double *omega_y_sin,
    const double *omega_y_cos,
    curandState *rand_states,
    const double kick_module,
    const double kick_sigma)
{
    // allocate memory for random kicks
    double *rand_kicks = (double *)malloc(4 * sizeof(double));

    double x;
    double px;
    double y;
    double py;
    double temp1;
    double temp2;
    unsigned int steps;

    size_t j = threadIdx.x + blockIdx.x * blockDim.x;

    if (j < n_samples)
    {
        // Load from shared
        x = g_x[j];
        px = g_px[j];
        y = g_y[j];
        py = g_py[j];
        steps = g_steps[j];

        // Track
        for (unsigned int k = 0; k < max_steps; ++k)
        {
            if (k == 0)
            {
                if (isnan(x) || isnan(px) || isnan(y) || isnan(py))
                {
                    break;
                }
            }
            temp1 = (px + x * x - y * y);
            temp2 = (py - 2 * x * y);
            
            if (mu != 0.0)
            {
                temp1 += mu * (x * x * x - 3 * y * y * x);
                temp2 -= mu * (3 * x * x * y - y * y * y);
            }

            px = - omega_x_sin[k] * x + omega_x_cos[k] * temp1;
            x  = + omega_x_cos[k] * x + omega_x_sin[k] * temp1;
            py = - omega_y_sin[k] * y + omega_y_cos[k] * temp2;
            y  = + omega_y_cos[k] * y + omega_y_sin[k] * temp2;
        
            // Generate random kick
            extract_random_kicks(
                rand_kicks, rand_states, j, kick_module, kick_sigma);
            x += rand_kicks[0];
            px += rand_kicks[1];
            y += rand_kicks[2];
            py += rand_kicks[3];

            if (x * x + y * y + px * px + py * py > barrier)
            {
                x = NAN;
                px = NAN;
                y = NAN;
                py = NAN;
                break;
            }
            steps += 1;
        }

        // Save in global
        g_x[j] = x;
        g_px[j] = px;
        g_y[j] = y;
        g_py[j] = py;
        g_steps[j] = steps;
    }
    // free memory
    free(rand_kicks);
}

__global__ void gpu_henon_track_inverse_with_kick(
    double *g_x,
    double *g_px,
    double *g_y,
    double *g_py,
    unsigned int *g_steps,
    const size_t n_samples,
    const unsigned int max_steps,
    const double barrier,
    const double mu,
    const double *omega_x_sin,
    const double *omega_x_cos,
    const double *omega_y_sin,
    const double *omega_y_cos,
    curandState *rand_states,
    const double kick_module,
    const double kick_sigma)
{
    // allocate memory for random kicks
    double *rand_kicks = (double *)malloc(4 * sizeof(double));

    double x;
    double px;
    double y;
    double py;
    double temp1;
    double temp2;
    unsigned int steps;

    size_t j = threadIdx.x + blockIdx.x * blockDim.x;

    if (j < n_samples)
    {
        // Load from shared
        x = g_x[j];
        px = g_px[j];
        y = g_y[j];
        py = g_py[j];
        steps = g_steps[j];

        // Track
        for (unsigned int k = max_steps; k != 0; --k)
        {
                if (isnan(x) || isnan(px) || isnan(y) || isnan(py))
                {
                    break;
                }

                temp1 = px;
                temp2 = py;
            
            px = +omega_x_sin[k - 1] * x + omega_x_cos[k - 1] * temp1;
            x = +omega_x_cos[k - 1] * x - omega_x_sin[k - 1] * temp1;
            py = +omega_y_sin[k - 1] * y + omega_y_cos[k - 1] * temp2;
            y = +omega_y_cos[k - 1] * y - omega_y_sin[k - 1] * temp2;

            px = (px - x * x + y * y);
            py = (py + 2 * x * y);
            
            if (mu != 0.0)
            {
                px -= mu * (x * x * x - 3 * y * y * x);
                py += mu * (3 * x * x * y - y * y * y);
            }

            // Generate random kick
            extract_random_kicks(
                rand_kicks, rand_states, j, kick_module, kick_sigma);
            x += rand_kicks[0];
            px += rand_kicks[1];
            y += rand_kicks[2];
            py += rand_kicks[3];
        
            if (x * x + y * y + px * px + py * py > barrier)
            {
                x = NAN;
                px = NAN;
                y = NAN;
                py = NAN;
                break;
            }
            steps -= 1;
        }

        // Save in global
        g_x[j] = x;
        g_px[j] = px;
        g_y[j] = y;
        g_py[j] = py;
        g_steps[j] = steps;
    }
    // free memory
    free(rand_kicks);
}

// PURE HENON CLASS IMPLEMENTATION

void henon::compute_a_modulation(unsigned int n_turns, bool inverse, std::string modulation_kind, double omega_0, double epsilon)
{
    // compute a modulation
    if (modulation_kind == "sps")
    {
        omega_x_vec = sps_modulation(
            omega_x, epsilon,
            !inverse ? global_steps : global_steps - n_turns + 1,
            !inverse ? global_steps + n_turns : global_steps + 1
        );
        omega_y_vec = sps_modulation(
            omega_y, epsilon, global_steps, global_steps + n_turns);
    }
    else if (modulation_kind == "basic")
    {
        assert(!std::isnan(omega_0));
        omega_x_vec = basic_modulation(
            omega_x, omega_0, epsilon,
            !inverse ? global_steps : global_steps - n_turns + 1,
            !inverse ? global_steps + n_turns : global_steps + 1
        );
        omega_y_vec = basic_modulation(
            omega_y, omega_0, epsilon,
            !inverse ? global_steps : global_steps - n_turns + 1,
            !inverse ? global_steps + n_turns : global_steps + 1
        );
    }
    else if (modulation_kind == "none")
    {
        // fill omega_x_vec with omega_x
        std::fill(omega_x_vec.begin(), omega_x_vec.end(), omega_x);
        // fill omega_y_vec with omega_y
        std::fill(omega_y_vec.begin(), omega_y_vec.end(), omega_y);
    }
    else if (modulation_kind == "gaussian")
    {
        omega_x_vec = gaussian_modulation(
            omega_x, epsilon,
            !inverse ? global_steps : global_steps - n_turns + 1,
            !inverse ? global_steps + n_turns : global_steps + 1
        );
        omega_y_vec = gaussian_modulation(
            omega_y, epsilon,
            !inverse ? global_steps : global_steps - n_turns + 1,
            !inverse ? global_steps + n_turns : global_steps + 1
        );
    }
    else if (modulation_kind == "uniform")
    {
        omega_x_vec = uniform_modulation(
            omega_x, epsilon,
            !inverse ? global_steps : global_steps - n_turns + 1,
            !inverse ? global_steps + n_turns : global_steps + 1
        );
        omega_y_vec = uniform_modulation(
            omega_y, epsilon,
            !inverse ? global_steps : global_steps - n_turns + 1,
            !inverse ? global_steps + n_turns : global_steps + 1
        );
    }
    else
    {
        throw std::runtime_error("Unknown modulation kind");
    }

    // copy to vectors
    omega_x_sin.resize(omega_x_vec.size());
    omega_x_cos.resize(omega_x_vec.size());
    omega_y_sin.resize(omega_y_vec.size());
    omega_y_cos.resize(omega_y_vec.size());

    for (size_t i = 0; i < omega_x_vec.size(); i++)
    {
        omega_x_sin[i] = sin(omega_x_vec[i]);
        omega_x_cos[i] = cos(omega_x_vec[i]);
        omega_y_sin[i] = sin(omega_y_vec[i]);
        omega_y_cos[i] = cos(omega_y_vec[i]);
    }   
}



henon::henon(const std::vector<double> &x_init,
             const std::vector<double> &px_init,
             const std::vector<double> &y_init,
             const std::vector<double> &py_init,
             double omega_x,
             double omega_y) : 
    omega_x(omega_x), omega_y(omega_y),
    x(x_init), y(y_init), px(px_init), py(py_init),
    x_0(x_init), y_0(y_init), px_0(px_init), py_0(py_init),
    steps(x_init.size(), 0), global_steps(0)
{
    // check if x_init, px_init, y_init and py_init are the same size
    assert(x_init.size() == px_init.size());
    assert(x_init.size() == y_init.size());
    assert(x_init.size() == py_init.size());

    // get number of cpu threads available
    n_threads_cpu = std::thread::hardware_concurrency();
}

void henon::reset()
{
    x = x_0;
    y = y_0;
    px = px_0;
    py = py_0;
    steps = std::vector<unsigned int>(x.size(), 0);
    global_steps = 0;
}

// getters
std::vector<double> henon::get_x() const { return x; }
std::vector<double> henon::get_px() const { return px; }
std::vector<double> henon::get_y() const { return y; }
std::vector<double> henon::get_py() const { return py; }

std::vector<double> henon::get_x0() const { return x_0; }
std::vector<double> henon::get_px0() const { return px_0; }
std::vector<double> henon::get_y0() const { return y_0; }
std::vector<double> henon::get_py0() const { return py_0; }

std::vector<unsigned int> henon::get_steps() const { return steps; }
unsigned int henon::get_global_steps() const { return global_steps; }
double henon::get_omega_x() const { return omega_x; }
double henon::get_omega_y() const { return omega_y; }

// setters
void henon::set_omega_x(double omega_x) { this->omega_x = omega_x; }
void henon::set_omega_y(double omega_y) { this->omega_y = omega_y; }

void henon::set_x(std::vector<double> x_init)
{
    // check if size is the same
    if (x.size() != x_init.size())
        throw std::invalid_argument("cpu_henon::set_x: vector sizes differ");
    x = x_init;
}
void henon::set_px(std::vector<double> px_init)
{
    // check if size is the same
    if (px.size() != px_init.size())
        throw std::invalid_argument("cpu_henon::set_px: vector sizes differ");
    px = px_init;
}
void henon::set_y(std::vector<double> y_init)
{
    // check if size is the same
    if (y.size() != y_init.size())
        throw std::invalid_argument("cpu_henon::set_y: vector sizes differ");
    y = y_init;
}
void henon::set_py(std::vector<double> py_init)
{
    // check if size is the same
    if (py.size() != py_init.size())
        throw std::invalid_argument("cpu_henon::set_py: vector sizes differ");
    assert(py_init.size() == py.size());
    py = py_init;
}
void henon::set_steps(std::vector<unsigned int> steps_init)
{
    // check if size is the same
    if (steps.size() != steps_init.size())
        throw std::invalid_argument("cpu_henon::set_steps: vector sizes differ");
    steps = steps_init;
}
void henon::set_steps(unsigned int steps_init)
{
    for (size_t i = 0; i < steps.size(); i++)
        steps[i] = steps_init;
}
void henon::set_global_steps(unsigned int global_steps_init)
{
    global_steps = global_steps_init;
}

std::array<std::vector<std::vector<double>>, 4> henon::full_track(unsigned int n_turns, double epsilon, double mu, double barrier, double kick_module, double kick_sigma, std::string modulation_kind, double omega_0)
{
    #ifdef PYBIND
        py::print("Computing modulation...");
    #endif
    // compute a modulation
    compute_a_modulation(n_turns, false, modulation_kind, omega_0, epsilon);

    #ifdef PYBIND
        py::print("Allocating vectors...");
    #endif
    // allocate 2d double vectors
    std::vector<std::vector<double>> x_vec(x.size(), std::vector<double>(n_turns+1, NAN));
    std::vector<std::vector<double>> px_vec(x.size(), std::vector<double>(n_turns+1, NAN));
    std::vector<std::vector<double>> y_vec(x.size(), std::vector<double>(n_turns+1, NAN));
    std::vector<std::vector<double>> py_vec(x.size(), std::vector<double>(n_turns+1, NAN));
    
    // fill first row of x_vec
    for (size_t i = 0; i < x.size(); i++)
    {
        x_vec[i][0] = x[i];
        px_vec[i][0] = px[i];
        y_vec[i][0] = y[i];
        py_vec[i][0] = py[i];
    }

    // update normal distribution
    bool kick_on = false;
    if (!isnan(kick_module) && !isnan(kick_sigma))
    {
        distribution = std::normal_distribution<double>(kick_module, kick_sigma);
        kick_on = true;
    }

    // for every element in vector x, execute cpu_henon_track in parallel
    std::vector<std::thread> threads;

    #ifdef PYBIND
        py::print("Starting threads...");
    #endif
    for (unsigned int i = 0; i < n_threads_cpu; i++)
    {
        threads.push_back(std::thread(
            [&](int thread_idx)
            {
                for (unsigned int j = thread_idx; j < x_vec.size(); j+=n_threads_cpu)
                {
                    cpu_full_henon_track(
                        j,
                        x_vec[j].data(), px_vec[j].data(),
                        y_vec[j].data(), py_vec[j].data(),
                        n_turns, barrier, mu,
                        omega_x_sin.data(), omega_x_cos.data(), 
                        omega_y_sin.data(), omega_y_cos.data(),
                        kick_on, &generator, distribution
                    );
                }
            },
            i
        ));
    }
    
    // join threads
    for (auto &t : threads)
        t.join();

    #ifdef PYBIND
        py::print("Returning results...");    
    #endif
    return {x_vec, px_vec, y_vec, py_vec};
}

std::vector<std::vector<double>> henon::full_track_and_lambda(unsigned int n_turns, double epsilon, double mu, double barrier, double kick_module, double kick_sigma, std::string modulation_kind, double omega_0,  std::function<std::vector<double>(std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>) > lambda)
{
    #ifdef PYBIND
        py::print("Computing modulation...");
    #endif
    // compute a modulation
    compute_a_modulation(n_turns, false, modulation_kind, omega_0, epsilon);

    // update normal distribution
    bool kick_on = false;
    if (!isnan(kick_module) && !isnan(kick_sigma))
    {
        distribution = std::normal_distribution<double>(kick_module, kick_sigma);
        kick_on = true;
    }

    // for every element in vector x, execute cpu_henon_track in parallel
    std::vector<std::thread> threads;

    #ifdef PYBIND
        py::print("Starting threads...");
    #endif

    std::vector<std::vector<double>> result_vec(x.size());

    for (unsigned int i = 0; i < n_threads_cpu; i++)
    {
        threads.push_back(std::thread(
            [&](int thread_idx)
            {
                std::vector<double> x_vec(n_turns + 1);
                std::vector<double> px_vec(n_turns + 1);
                std::vector<double> y_vec(n_turns + 1);
                std::vector<double> py_vec(n_turns + 1);
                for (unsigned int j = thread_idx; j < x.size(); j+=n_threads_cpu)
                {
                    x_vec[0] = x[j];
                    px_vec[0] = px[j];
                    y_vec[0] = y[j];
                    py_vec[0] = py[j];
                    cpu_full_henon_track(
                        j,
                        x_vec.data(), px_vec.data(),
                        y_vec.data(), py_vec.data(),
                        n_turns, barrier, mu,
                        omega_x_sin.data(), omega_x_cos.data(), 
                        omega_y_sin.data(), omega_y_cos.data(),
                        kick_on, &generator, distribution
                    );
                    auto result = lambda(x_vec, px_vec, y_vec, py_vec);
                    result_vec[j] = result;
                }
            },
            i
        ));
    }
    
    // join threads
    for (auto &t : threads)
        t.join();

    #ifdef PYBIND
        py::print("Returning results...");    
    #endif

    return result_vec;
}

std::vector<std::vector<double>> henon::birkhoff_tunes(unsigned int n_turns, double epsilon, double mu, double barrier, double kick_module, double kick_sigma, std::string modulation_kind, double omega_0, std::vector<unsigned int> from, std::vector<unsigned int> to)
{
    auto lambda = [&](std::vector<double> const &x_vec, std::vector<double> const &px_vec, std::vector<double> const &y_vec, std::vector<double> const &py_vec)
    {
        return birkhoff_tune_vec(x_vec, px_vec, y_vec, py_vec, from, to);
    };
    return full_track_and_lambda(n_turns, epsilon, mu, barrier, kick_module, kick_sigma, modulation_kind, omega_0, lambda);
}


std::vector<std::vector<double>> henon::fft_tunes(unsigned int n_turns, double epsilon, double mu, double barrier, double kick_module, double kick_sigma, std::string modulation_kind, double omega_0, std::vector<unsigned int> from, std::vector<unsigned int> to)
{
    auto lambda = [&](std::vector<double> const &x_vec, std::vector<double> const &px_vec, std::vector<double> const &y_vec, std::vector<double> const &py_vec)
    {
        return fft_tune_vec(x_vec, px_vec, y_vec, py_vec, from, to);
    };
    return full_track_and_lambda(n_turns, epsilon, mu, barrier, kick_module, kick_sigma, modulation_kind, omega_0, lambda);
}

// GPU HENON CLASS DERIVATIVE

gpu_henon::gpu_henon(const std::vector<double> &x_init,
              const std::vector<double> &px_init,
              const std::vector<double> &y_init,
              const std::vector<double> &py_init,
              double omega_x,
              double omega_y) : 
    henon(x_init, px_init, y_init, py_init, omega_x, omega_y)
{
    // load vectors on gpu
    cudaMalloc(&d_x, x.size() * sizeof(double));
    cudaMalloc(&d_px, px.size() * sizeof(double));
    cudaMalloc(&d_y, y.size() * sizeof(double));
    cudaMalloc(&d_py, py.size() * sizeof(double));
    cudaMalloc(&d_steps, steps.size() * sizeof(unsigned int));

    // copy vectors to gpu
    cudaMemcpy(d_x, x.data(), x.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_px, px.data(), px.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y.data(), y.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_py, py.data(), py.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_steps, steps.data(), steps.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // compute number of blocks and threads
    n_samples = x.size();
    n_threads = NT;
    n_blocks = (n_samples + n_threads - 1) / n_threads;

    // initialize curandom states
    cudaMalloc(&d_rand_states, n_threads * n_blocks * sizeof(curandState));
    setup_random_states<<<n_blocks, n_threads>>>(d_rand_states, clock());
}

gpu_henon::~gpu_henon()
{
    cudaFree(d_x);
    cudaFree(d_px);
    cudaFree(d_y);
    cudaFree(d_py);
    cudaFree(d_steps);
    cudaFree(d_rand_states);
}

void gpu_henon::reset()
{
    // copy initial values to host vectors
    x = x_0;
    y = y_0;
    px = px_0;
    py = py_0;
    steps = std::vector<unsigned int>(x.size(), 0);
    global_steps = 0;

    // copy initial values to gpu
    cudaMemcpy(d_x, x.data(), x.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_px, px.data(), px.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y.data(), y.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_py, py.data(), py.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_steps, steps.data(), steps.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // initialize curandom states
    setup_random_states<<<n_blocks, n_threads>>>(d_rand_states, clock());
}

void gpu_henon::track(unsigned int n_turns, double epsilon, double mu, double barrier, double kick_module, double kick_sigma, bool inverse, std::string modulation_kind, double omega_0)
{
    // compute a modulation
    compute_a_modulation(n_turns, inverse, modulation_kind, omega_0, epsilon);

    // copy to vectors
    omega_x_sin.resize(omega_x_vec.size());
    omega_x_cos.resize(omega_x_vec.size());
    omega_y_sin.resize(omega_y_vec.size());
    omega_y_cos.resize(omega_y_vec.size());

    for (size_t i = 0; i < omega_x_vec.size(); i++)
    {
        omega_x_sin[i] = sin(omega_x_vec[i]);
        omega_x_cos[i] = cos(omega_x_vec[i]);
        omega_y_sin[i] = sin(omega_y_vec[i]);
        omega_y_cos[i] = cos(omega_y_vec[i]);
    }

    // copy to gpu
    cudaMalloc(&d_omega_x_sin, omega_x_sin.size() * sizeof(double));
    cudaMalloc(&d_omega_x_cos, omega_x_cos.size() * sizeof(double));
    cudaMalloc(&d_omega_y_sin, omega_y_sin.size() * sizeof(double));
    cudaMalloc(&d_omega_y_cos, omega_y_cos.size() * sizeof(double));

    cudaMemcpy(d_omega_x_sin, omega_x_sin.data(), omega_x_sin.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_omega_x_cos, omega_x_cos.data(), omega_x_cos.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_omega_y_sin, omega_y_sin.data(), omega_y_sin.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_omega_y_cos, omega_y_cos.data(), omega_y_cos.size() * sizeof(double), cudaMemcpyHostToDevice);

    // run the simulation
    if (std::isnan(kick_module) || std::isnan(kick_sigma))
    {
        if (!inverse)
            gpu_henon_track<<<n_blocks, n_threads>>>(
                d_x, d_px, d_y, d_py, d_steps,
                n_samples, n_turns, barrier, mu,
                d_omega_x_sin, d_omega_x_cos, d_omega_y_sin, d_omega_y_cos);
        else
            gpu_henon_track_inverse<<<n_blocks, n_threads>>>(
                d_x, d_px, d_y, d_py, d_steps,
                n_samples, n_turns, barrier, mu,
                d_omega_x_sin, d_omega_x_cos, d_omega_y_sin, d_omega_y_cos);
    }
    else
    {
        if (!inverse)
            gpu_henon_track_with_kick<<<n_blocks, n_threads>>>(
                d_x, d_px, d_y, d_py, d_steps,
                n_samples, n_turns, barrier, mu,
                d_omega_x_sin, d_omega_x_cos, d_omega_y_sin, d_omega_y_cos,
                d_rand_states, kick_module, kick_sigma);
        else
            gpu_henon_track_inverse_with_kick<<<n_blocks, n_threads>>>(
                d_x, d_px, d_y, d_py, d_steps,
                n_samples, n_turns, barrier, mu,
                d_omega_x_sin, d_omega_x_cos, d_omega_y_sin, d_omega_y_cos,
                d_rand_states, kick_module, kick_sigma);
    }
    // check for cuda errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }

    // copy back to host
    cudaMemcpy(x.data(), d_x, n_samples * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(px.data(), d_px, n_samples * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(y.data(), d_y, n_samples * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(py.data(), d_py, n_samples * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(steps.data(), d_steps, n_samples * sizeof(int), cudaMemcpyDeviceToHost);

    // clear the omega vectors
    omega_x_sin.clear();
    omega_x_cos.clear();
    omega_y_sin.clear();
    omega_y_cos.clear();

    // clear the gpu omega vectors
    cudaFree(d_omega_x_sin);
    cudaFree(d_omega_x_cos);
    cudaFree(d_omega_y_sin);
    cudaFree(d_omega_y_cos);

    // Update the counter
    if (!inverse)
        global_steps += n_turns;
    else
        global_steps -= n_turns;
}

// setters

void gpu_henon::set_x(std::vector<double> x)
{
    // check if size is correct
    if (x.size() != n_samples)
        throw std::runtime_error("The size of the x vector is not correct.");
    this->x = x;
    // copy to gpu
    cudaMemcpy(d_x, x.data(), x.size() * sizeof(double), cudaMemcpyHostToDevice);
}
void gpu_henon::set_px(std::vector<double> px)
{
    // check if size is correct
    if (px.size() != n_samples)
        throw std::runtime_error("The size of the px vector is not correct.");
    this->px = px;
    // copy to gpu
    cudaMemcpy(d_px, px.data(), px.size() * sizeof(double), cudaMemcpyHostToDevice);
}
void gpu_henon::set_y(std::vector<double> y)
{
    // check if size is correct
    if (y.size() != n_samples)
        throw std::runtime_error("The size of the y vector is not correct.");
    this->y = y;
    // copy to gpu
    cudaMemcpy(d_y, y.data(), y.size() * sizeof(double), cudaMemcpyHostToDevice);
}
void gpu_henon::set_py(std::vector<double> py)
{
    // check if size is correct
    if (py.size() != n_samples)
        throw std::runtime_error("The size of the py vector is not correct.");
    this->py = py;
    // copy to gpu
    cudaMemcpy(d_py, py.data(), py.size() * sizeof(double), cudaMemcpyHostToDevice);
}
void gpu_henon::set_steps(std::vector<unsigned int> steps)
{
    // check if size is correct
    if (steps.size() != n_samples)
        throw std::runtime_error("The size of the steps vector is not correct.");
    this->steps = steps;
    // copy to gpu
    cudaMemcpy(d_steps, steps.data(), steps.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
}
void gpu_henon::set_steps(unsigned int unique_step)
{
    // apply unique_step to entire steps vector
    for (unsigned int i = 0; i < n_samples; i++)
        steps[i] = unique_step;
    // copy to gpu
    cudaMemcpy(d_steps, steps.data(), steps.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
}

// CPU HENON CLASS DERIVATIVE

cpu_henon::cpu_henon(const std::vector<double> &x_init,
                     const std::vector<double> &px_init,
                     const std::vector<double> &y_init,
                     const std::vector<double> &py_init,
                     double omega_x,
                     double omega_y) : 
    henon(x_init, px_init, y_init, py_init, omega_x, omega_y)
{}

cpu_henon::~cpu_henon() {}

void cpu_henon::track(unsigned int n_turns, double epsilon, double mu, double barrier, double kick_module, double kick_sigma, bool inverse, std::string modulation_kind, double omega_0)
{
    // compute a modulation
    compute_a_modulation(n_turns, inverse, modulation_kind, omega_0, epsilon);

    // copy to vectors
    omega_x_sin.resize(omega_x_vec.size());
    omega_x_cos.resize(omega_x_vec.size());
    omega_y_sin.resize(omega_y_vec.size());
    omega_y_cos.resize(omega_y_vec.size());

    for (size_t i = 0; i < omega_x_vec.size(); i++)
    {
        omega_x_sin[i] = sin(omega_x_vec[i]);
        omega_x_cos[i] = cos(omega_x_vec[i]);
        omega_y_sin[i] = sin(omega_y_vec[i]);
        omega_y_cos[i] = cos(omega_y_vec[i]);
    }

    // update normal distribution
    bool kick_on = false;
    if (!isnan(kick_module) && !isnan(kick_sigma))
    {
        distribution = std::normal_distribution<double>(kick_module, kick_sigma);
        kick_on = true;
    }


    // for every element in vector x, execute cpu_henon_track in parallel
    std::vector<std::thread> threads;

    for (unsigned int i = 0; i < n_threads_cpu; i++)
    {
        threads.push_back(std::thread(
            [&](int thread_idx)
            {
                for (unsigned int j = thread_idx; j < x.size(); j+=n_threads_cpu)
                {
                    cpu_henon_track(
                        j, x.data(), px.data(), y.data(), py.data(), steps.data(),
                        n_turns, barrier, mu, 
                        omega_x_sin.data(), omega_x_cos.data(), 
                        omega_y_sin.data(), omega_y_cos.data(), 
                        inverse, kick_on, &generator, distribution
                    );
                }
            }, 
            i
        ));
    }
    
    // join threads
    for (auto &t : threads)
        t.join();
    
    // update global_steps
    if (!inverse)
        global_steps += n_turns;
    else
        global_steps -= n_turns;
}