#include "henon.h"

#ifdef PYBIND
    namespace py = pybind11;
#endif

void check_keyboard_interrupt(){
#ifdef PYBIND
    if (PyErr_CheckSignals() != 0)
    {
        std::cout << "Keyboard interrupt" << std::endl;
        throw py::error_already_set();
    }
#endif
}

void check_cuda_errors() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    check_keyboard_interrupt();
}


__host__ __device__ bool check_barrier(const double &x, const double &px, const double &y, const double &py, const double &barrier_pow_2)
{
    return (x*x + px*px) + (y*y + py*py) < barrier_pow_2;
}

__host__ __device__ bool check_nan(const double &x, const double &px, const double &y, const double &py)
{
    return isnan(x) || isnan(px) || isnan(y) || isnan(py);
}

__host__ __device__ void set_to_nan(double &x, double &px, double &y, double &py)
{
    x = NAN;
    px = NAN;
    y = NAN;
    py = NAN;
}

__host__ __device__ void sextupole(double &x, double &px, double &y, double &py, const bool &reverse)
{
    if (!reverse)
    {
        px += (x) * (x) - (y) * (y);
        py += -2.0 * (x) * (y);
    }
    else
    {
        px -= (x) * (x) - (y) * (y);
        py -= -2.0 * (x) * (y);
    }
}

__host__ __device__ void octupole(double &x, double &px, double &y, double &py, const double &mu, const bool &reverse)
{
    if (!reverse)
    {
        px += mu * ((x) * (x) * (x) - 3.0 * (y) * (y) * (x));
        py += mu * ((y) * (y) * (y) - 3.0 * (x) * (x) * (y));
    }
    else
    {
        px -= mu * ((x) * (x) * (x) - 3.0 * (y) * (y) * (x));
        py -= mu * ((y) * (y) * (y) - 3.0 * (x) * (x) * (y));
    }
}

__host__ __device__ void rotation(double &x, double &px, const double &sin, const double &cos, const bool &reverse)
{
    double x_tmp = x;
    double px_tmp = px;
    if (!reverse)
    {
        x = x_tmp * (cos) + px_tmp * (sin);
        px = - x_tmp * (sin) + px_tmp * (cos);
    }
    else
    {
        x = x_tmp * (cos) - px_tmp * (sin);
        px = x_tmp * (sin) + px_tmp * (cos);
    }
}

__host__ __device__ std::array<std::array<double, 4>, 4> tangent_matrix(const double &x, const double &px, const double &y, const double &py, const double &sx, const double &cx, const double &sy, const double &cy, const double &mu)
{
    std::array<std::array<double, 4>, 4> matrix;
    matrix[0][0] = cx + sx * (2 * x + mu * 3 * x * x - mu * 3 * y * y);
    matrix[0][1] = sx;
    matrix[0][2] = sx * (-2 * y - mu * 6 * x * y);
    matrix[0][3] = 0.0;

    matrix[1][0] = -sx + cx * (2 * x + mu * 3 * x * x - mu * 3 * y * y);
    matrix[1][1] = cx;
    matrix[1][2] = cx * (-2 * y - mu * 6 * x * y);
    matrix[1][3] = 0.0;

    matrix[2][0] = sy * (-2 * y - mu * 6 * x * y);
    matrix[2][1] = 0.0;
    matrix[2][2] = cy + sy * (-2 * x - mu * 3 * x * x + mu * 3 * y * y);
    matrix[2][3] = sy;

    matrix[3][0] = cy * (-2 * y - mu * 6 * x * y);
    matrix[3][1] = 0.0;
    matrix[3][2] = -sy + cy * (-2 * x - mu * 3 * x * x + mu * 3 * y * y);
    matrix[3][3] = cy;

    return matrix;
}

__host__ __device__ std::array<std::array<double, 4>, 4> multiply_matrix(const double *m2, const std::array<std::array<double, 4>, 4> &m1)
{
    std::array<std::array<double, 4>, 4> m3;
    for (unsigned int i = 0; i < 4; i++)
    {
        for (unsigned int j = 0; j < 4; j++)
        {
            m3[i][j] = 0.0;
            for (unsigned int k = 0; k < 4; k++)
            {
                m3[i][j] += m1[i][k] * m2[k * 4 + j];
            }
        }
    }
    return m3;
}

__host__ __device__ double tr_transpose_matrix(const double *x)
{
    double result = 0.0;
    for (int i = 0; i < 16; i++)
    {
        result += x[i] * x[i];
    }
    return result;
}

__host__ void random_4d_kick(double &x, double &px, double &y, double &py, const double &kick_module, std::mt19937_64 &generator)
{
    // create random uniform distribution between -1 and 1
    auto uni_dist = std::uniform_real_distribution<double>(-1.0, 1.0);
    // create the array
    std::array<double, 4> kick;
    // fill it with random uniform numbers between -1 and 1
    for (auto &kick_component : kick)
    {
        kick_component = uni_dist(generator);
    }
    while (kick[0] * kick[0] + kick[1] * kick[1] > 1.0)
    {
        kick[0] = uni_dist(generator);
        kick[1] = uni_dist(generator);
    }
    while (kick[2] * kick[2] + kick[3] * kick[3] > 1.0)
    {
        kick[2] = uni_dist(generator);
        kick[3] = uni_dist(generator);
    }
    // extract the module from the normal distribution
    auto kick_fix = (1 - kick[0] * kick[0] - kick[1] * kick[1]) / (kick[2] * kick[2] + kick[3] * kick[3]);
    // scale the kick
    x  += kick[0] * kick_module;
    px += kick[1] * kick_module;
    y  += kick[2] * kick_fix * kick_module;
    py += kick[3] * kick_fix * kick_module;
}

__device__ void random_4d_kick(double &x, double &px, double &y, double &py, const double &kick_module, curandState &state)
{
    double rand_kicks[4];
    rand_kicks[0] = curand_uniform(&state) * 2.0 - 1.0;
    rand_kicks[1] = curand_uniform(&state) * 2.0 - 1.0;
    while (rand_kicks[0] * rand_kicks[0] + rand_kicks[1] * rand_kicks[1] > 1.0)
    {
        rand_kicks[0] = curand_uniform(&state) * 2.0 - 1.0;
        rand_kicks[1] = curand_uniform(&state) * 2.0 - 1.0;
    }
    rand_kicks[2] = curand_uniform(&state) * 2.0 - 1.0;
    rand_kicks[3] = curand_uniform(&state) * 2.0 - 1.0;
    while (rand_kicks[2] * rand_kicks[2] + rand_kicks[3] * rand_kicks[3] > 1.0)
    {
        rand_kicks[2] = curand_uniform(&state) * 2.0 - 1.0;
        rand_kicks[3] = curand_uniform(&state) * 2.0 - 1.0;
    }
    double rfix = (1 - rand_kicks[0] * rand_kicks[0] - rand_kicks[1] * rand_kicks[1]) / (rand_kicks[2] * rand_kicks[2] + rand_kicks[3] * rand_kicks[3]);
    
    x  += rand_kicks[0] * kick_module;
    px += rand_kicks[1] * kick_module;
    y  += rand_kicks[2] * rfix * kick_module;
    py += rand_kicks[3] * rfix * kick_module;
}

__host__ __device__ double displacement(const double &x1, const double &px1, const double &y1, const double &py1, const double &x2, const double &px2, const double &y2, const double &py2)
{
    return sqrt((x1 - x2) * (x1 - x2) + (px1 - px2) * (px1 - px2) + (y1 - y2) * (y1 - y2) + (py1 - py2) * (py1 - py2));
}

__host__ __device__ void realign(double &x, double &px, double &y, double &py, const double &x_center, const double &px_center, const double &y_center, const double &py_center, const double &initial_module, const double &final_module)
{
    x = x_center + (x - x_center) * (final_module / initial_module);
    px = px_center + (px - px_center) * (final_module / initial_module);
    y = y_center + (y - y_center) * (final_module / initial_module);
    py = py_center + (py - py_center) * (final_module / initial_module);
}

__host__ void henon_step(
    double &x, double &px, double &y, double &py,
    unsigned int &steps,
    const double *sinx, const double *cosx,
    const double *siny, const double *cosy,
    const double &barrier_pow_2,
    const double &mu,
    const double &kick_module,
    std::mt19937_64 &generator,
    const bool reverse)
{
    if (check_nan(x, px, y, py))
    {
        return;
    }

    if (!reverse)
    {
        if (!isnan(kick_module))
        {
            random_4d_kick(x, px, y, py, kick_module, generator);
        }

        sextupole(x, px, y, py, reverse);

        if (mu != 0.0)
        {
            octupole(x, px, y, py, mu, reverse);
        }

        rotation(x, px, sinx[steps], cosx[steps], reverse);
        rotation(y, py, siny[steps], cosy[steps], reverse);
    }
    else
    {
        rotation(x, px, sinx[steps - 1], cosx[steps - 1], reverse);
        rotation(y, py, siny[steps - 1], cosy[steps - 1], reverse);

        if (mu != 0.0)
        {
            octupole(x, px, y, py, mu, reverse);
        }

        sextupole(x, px, y, py, reverse);

        if (!isnan(kick_module))
        {
            random_4d_kick(x, px, y, py, kick_module, generator);
        }
    }

    if (check_barrier(x, px, y, py, barrier_pow_2))
    {
        steps += (reverse ? -1 : 1);
        return;
    }
    else
    {
        set_to_nan(x, px, y, py);
    }
}

__device__ void henon_step(
    double &x, double &px, double &y, double &py,
    unsigned int &steps,
    const double *sinx, const double *cosx,
    const double *siny, const double *cosy,
    const double &barrier_pow_2,
    const double &mu,
    const double &kick_module,
    curandState &state,
    const bool reverse)
{
    if (check_nan(x, px, y, py))
    {
        return;
    }

    if (!reverse)
    {
        if (!isnan(kick_module))
        {
            random_4d_kick(x, px, y, py, kick_module, state);
        }

        sextupole(x, px, y, py, reverse);

        if (mu != 0.0)
        {
            octupole(x, px, y, py, mu, reverse);
        }

        rotation(x, px, sinx[steps], cosx[steps], reverse);
        rotation(y, py, siny[steps], cosy[steps], reverse);
    }
    else
    {
        rotation(x, px, sinx[steps - 1], cosx[steps - 1], reverse);
        rotation(y, py, siny[steps - 1], cosy[steps - 1], reverse);

        if (mu != 0.0)
        {
            octupole(x, px, y, py, mu, reverse);
        }

        sextupole(x, px, y, py, reverse);

        if (!isnan(kick_module))
        {
            random_4d_kick(x, px, y, py, kick_module, state);
        }
    }

    if (check_barrier(x, px, y, py, barrier_pow_2))
    {
        steps += (reverse ? -1 : 1);
        return;
    }
    else
    {
        set_to_nan(x, px, y, py);
    }
}

__global__ void gpu_compute_displacements(
    double *x1, double *px1, double *y1, double *py1,
    double *out, const size_t size)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size)
    {
        return;
    }

    if (check_nan(x1[i], px1[i], y1[i], py1[i]) || check_nan(x1[i + size], px1[i + size], y1[i + size], py1[i + size]))
    {
        out[i] = NAN;
        return;
    }

    double x2 = x1[i + size];
    double px2 = px1[i + size];
    double y2 = y1[i + size];
    double py2 = py1[i + size];

    out[i] = displacement(x1[i], px1[i], y1[i], py1[i], x2, px2, y2, py2);
}

__global__ void gpu_compute_displacements_and_realign(
    double *x1, double *px1, double *y1, double *py1,
    double *out, const size_t size, const double low_module, const double limit_module)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size)
    {
        return;
    }

    if (check_nan(x1[i], px1[i], y1[i], py1[i]) || check_nan(x1[i + size], px1[i + size], y1[i + size], py1[i + size]))
    {
        out[i] = NAN;
        return;
    }
    
    double tmp_displacement = displacement(x1[i], px1[i], y1[i], py1[i], x1[i + size], px1[i + size], y1[i + size], py1[i + size]);
    if (tmp_displacement > limit_module)
    {
        out[i] += tmp_displacement - low_module;
        realign(
            x1[i + size], px1[i + size], y1[i + size], py1[i + size],
            x1[i], px1[i], y1[i], py1[i], tmp_displacement, low_module);
    }
}

__global__ void gpu_add_to_ratio(
    double *new_displacement,
    double *old_displacement,
    double *ratio,
    const unsigned int step,
    const size_t n_samples)
{
    size_t j = threadIdx.x + blockIdx.x * blockDim.x;
    if (j >= n_samples)
    {
        return;
    }

    if (isnan(old_displacement[j]) || isnan(new_displacement[j]))
    {
        ratio[j] = NAN;
    }
    else
    {
        ratio[j] += step * log10(new_displacement[j] / old_displacement[j]);
    }
}

__global__ void gpu_sum_two_arrays(
    double *in1,
    const double *in2,
    const size_t size)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size)
    {
        return;
    }

    in1[i] += in2[i];
}

__global__ void gpu_henon_track(
    double *g_x,
    double *g_px,
    double *g_y,
    double *g_py,
    unsigned int *g_steps,
    const size_t n_samples,
    const unsigned int max_steps,
    const double barrier_pow_2,
    const double mu,
    const double *omega_x_sin,
    const double *omega_x_cos,
    const double *omega_y_sin,
    const double *omega_y_cos,
    const double kick_module,
    curandState *state,
    const bool reverse)
{
    double x;
    double px;
    double y;
    double py;
    unsigned int steps;

    size_t j = threadIdx.x + blockIdx.x * blockDim.x;

    if (j < n_samples)
    {
        // Load from global
        x = g_x[j];
        px = g_px[j];
        y = g_y[j];
        py = g_py[j];
        steps = g_steps[j];

        // Track
        for (unsigned int k = 0; k < max_steps; ++k)
        {
            henon_step(
                x, px, y, py,
                steps,
                omega_x_sin, omega_x_cos,
                omega_y_sin, omega_y_cos,
                barrier_pow_2,
                mu,
                kick_module, state[j], reverse);
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


// PURE HENON CLASS IMPLEMENTATION

void henon::compute_a_modulation(unsigned int n_turns, double omega_x, double omega_y, std::string modulation_kind, double omega_0, double epsilon, unsigned int offset)
{
    // compute a modulation
    tie(omega_x_vec, omega_y_vec) = pick_a_modulation(n_turns, omega_x, omega_y, modulation_kind, omega_0, epsilon, offset);

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

    allowed_steps = n_turns;
}

henon::henon(const std::vector<double> &x_init,
             const std::vector<double> &px_init,
             const std::vector<double> &y_init,
             const std::vector<double> &py_init) : 
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

    // create default vector of modulations
    this->compute_a_modulation(256, 0.168, 0.201, "none", NAN, NAN, 0);
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

// setters

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
    if (steps_init > allowed_steps)
        throw std::invalid_argument("cpu_henon::set_steps: steps_init > allowed_steps");
    for (size_t i = 0; i < steps.size(); i++)
        steps[i] = steps_init;
}
void henon::set_global_steps(unsigned int global_steps_init)
{
    if (global_steps_init > allowed_steps)
        throw std::invalid_argument("cpu_henon::set_global_steps: global_steps_init > allowed_steps");
    global_steps = global_steps_init;
}

std::array<std::vector<std::vector<double>>, 4> henon::full_track(unsigned int n_turns, double mu, double barrier, double kick_module)
{
    // check if n_turns is allowed
    if (n_turns + global_steps > allowed_steps)
        throw std::invalid_argument("cpu_henon::full_track: n_turns is too large");

    std::cout<< "Allocating vectors..." << std::endl;

    // allocate 2d double vectors
    std::vector<std::vector<double>> x_vec(x.size(), std::vector<double>(n_turns+1, NAN));
    std::vector<std::vector<double>> px_vec(x.size(), std::vector<double>(n_turns+1, NAN));
    std::vector<std::vector<double>> y_vec(x.size(), std::vector<double>(n_turns+1, NAN));
    std::vector<std::vector<double>> py_vec(x.size(), std::vector<double>(n_turns+1, NAN));
    
    double barrier_pow_2 = barrier * barrier;

    // fill first row of x_vec
    for (size_t i = 0; i < x.size(); i++)
    {
        x_vec[i][0] = x[i];
        px_vec[i][0] = px[i];
        y_vec[i][0] = y[i];
        py_vec[i][0] = py[i];
    }

    // for every element in vector x, execute cpu_henon_track in parallel
    std::vector<std::thread> threads;

    std::cout << "Starting threads..." << std::endl;

    for (unsigned int i = 0; i < n_threads_cpu; i++)
    {
        threads.push_back(std::thread(
            [&](int thread_idx)
            {
                std::mt19937_64 rng;
                for (unsigned int j = thread_idx; j < x_vec.size(); j += n_threads_cpu)
                {
                    for (unsigned int k = 1; k < n_turns + 1; k++)
                    {
                        henon_step(
                            x[j], px[j], y[j], py[j], steps[j],
                            omega_x_sin.data(), omega_x_cos.data(), 
                            omega_y_sin.data(), omega_y_cos.data(),
                            barrier_pow_2, mu, kick_module,
                            rng, false
                        );
                        
                        x_vec[j][k] = x[j];
                        px_vec[j][k] = px[j];
                        y_vec[j][k] = y[j];
                        py_vec[j][k] = py[j];
                    }
                }
            },
            i
        ));
    }
    
    // join threads
    for (auto &t : threads)
        t.join();

    std::cout << "Returning results..." << std::endl;

    global_steps += n_turns;
    return {x_vec, px_vec, y_vec, py_vec};
}

std::vector<std::vector<double>> henon::full_track_and_lambda(unsigned int n_turns, double mu, double barrier, double kick_module, std::function<std::vector<double>(std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>) > lambda)
{
    // check if n_turns is allowed
    if (n_turns + global_steps > allowed_steps)
        throw std::invalid_argument("cpu_henon::full_track: n_turns is too large");

    double barrier_pow_2 = barrier * barrier;

    // for every element in vector x, execute cpu_henon_track in parallel
    std::vector<std::thread> threads;

    std::cout << "Starting threads..." << std::endl;
    
    std::vector<std::vector<double>> result_vec(x.size());
    for (unsigned int i = 0; i < n_threads_cpu; i++)
    {
        threads.push_back(std::thread(
            [&](int thread_idx)
            {
                std::mt19937_64 rng;
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
                    for (unsigned int k = 1; k < n_turns + 1; k++)
                    {
                        henon_step(
                            x[j], px[j], y[j], py[j], steps[j],
                            omega_x_sin.data(), omega_x_cos.data(),
                            omega_y_sin.data(), omega_y_cos.data(),
                            barrier_pow_2, mu, kick_module,
                            rng, false);

                        x_vec[k] = x[j];
                        px_vec[k] = px[j];
                        y_vec[k] = y[j];
                        py_vec[k] = py[j];
                    }
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

    std::cout << "Returning results..." << std::endl;    

    global_steps += n_turns;
    return result_vec;
}

std::vector<std::vector<double>> henon::birkhoff_tunes(unsigned int n_turns, double mu, double barrier, double kick_module, std::vector<unsigned int> from, std::vector<unsigned int> to)
{
    auto lambda = [&](std::vector<double> const &x_vec, std::vector<double> const &px_vec, std::vector<double> const &y_vec, std::vector<double> const &py_vec)
    {
        return birkhoff_tune_vec(x_vec, px_vec, y_vec, py_vec, from, to);
    };
    return full_track_and_lambda(n_turns, mu, barrier, kick_module, lambda);
}


std::vector<std::vector<double>> henon::fft_tunes(unsigned int n_turns, double mu, double barrier, double kick_module, std::vector<unsigned int> from, std::vector<unsigned int> to)
{
     // check if n_turns is allowed
    if (n_turns + global_steps > allowed_steps)
        throw std::invalid_argument("cpu_henon::full_track: n_turns is too large");

    double barrier_pow_2 = barrier * barrier;

    std::vector<unsigned int> n_list;
    n_list.push_back(n_turns);
    for (unsigned int i = 0; i < from.size(); i++)
    {
        if (from[i] > to[i])
            throw std::invalid_argument("cpu_henon::fft_tunes: from[i] > to[i]");
        n_list.push_back(to[i] - from[i] + 1);
    }
    // remove repeteated elements
    std::sort(n_list.begin(), n_list.end());
    n_list.erase(std::unique(n_list.begin(), n_list.end()), n_list.end());

    // for every element in vector x, execute cpu_henon_track in parallel
    std::vector<std::thread> threads;

    std::cout << "Starting threads..." << std::endl;

    std::vector<std::vector<double>> result_vec(x.size());
    for (unsigned int i = 0; i < n_threads_cpu; i++)
    {
        threads.push_back(std::thread(
            [&](int thread_idx)
            {
                // create plans
                auto plans = fft_allocate(n_list);

                std::mt19937_64 rng;
                std::vector<double> x_vec(n_turns + 1);
                std::vector<double> px_vec(n_turns + 1);
                std::vector<double> y_vec(n_turns + 1);
                std::vector<double> py_vec(n_turns + 1);
                for (unsigned int j = thread_idx; j < x.size(); j += n_threads_cpu)
                {
                    x_vec[0] = x[j];
                    px_vec[0] = px[j];
                    y_vec[0] = y[j];
                    py_vec[0] = py[j];
                    for (unsigned int k = 1; k < n_turns + 1; k++)
                    {
                        henon_step(
                            x[j], px[j], y[j], py[j], steps[j],
                            omega_x_sin.data(), omega_x_cos.data(),
                            omega_y_sin.data(), omega_y_cos.data(),
                            barrier_pow_2, mu, kick_module,
                            rng, false);

                        x_vec[k] = x[j];
                        px_vec[k] = px[j];
                        y_vec[k] = y[j];
                        py_vec[k] = py[j];
                    }
                    auto result = fft_tune_vec(x_vec, px_vec, y_vec, py_vec, from, to, plans);
                    result_vec[j] = result;
                }
                // free plans
                fft_free(plans);
            },
            i));
    }

    // join threads
    for (auto &t : threads)
        t.join();

    std::cout << "Returning results..." << std::endl;

    global_steps += n_turns;
    return result_vec;
}

// GPU HENON CLASS DERIVATIVE

gpu_henon::gpu_henon(const std::vector<double> &x_init,
              const std::vector<double> &px_init,
              const std::vector<double> &y_init,
              const std::vector<double> &py_init) : 
    henon(x_init, px_init, y_init, py_init)
{
    std::cout << "Initializing GPU henon..." << std::endl;

    // load vectors on gpu
    cudaMalloc(&d_x, x.size() * sizeof(double));
    cudaMalloc(&d_px, px.size() * sizeof(double));
    cudaMalloc(&d_y, y.size() * sizeof(double));
    cudaMalloc(&d_py, py.size() * sizeof(double));
    cudaMalloc(&d_steps, steps.size() * sizeof(unsigned int));
    check_cuda_errors();

    cudaMalloc(&d_omega_x_sin, 256 * sizeof(double));
    cudaMalloc(&d_omega_x_cos, 256 * sizeof(double));
    cudaMalloc(&d_omega_y_sin, 256 * sizeof(double));
    cudaMalloc(&d_omega_y_cos, 256 * sizeof(double));
    check_cuda_errors();

    // copy vectors to gpu
    cudaMemcpy(d_x, x.data(), x.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_px, px.data(), px.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y.data(), y.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_py, py.data(), py.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_steps, steps.data(), steps.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // fill d_omega_x_sin and d_omega_x_cos with values
    cudaMemset(d_omega_x_sin, sin(M_PI / 6), 256 * sizeof(double));
    cudaMemset(d_omega_x_cos, cos(M_PI / 6), 256 * sizeof(double));
    cudaMemset(d_omega_y_sin, sin(M_PI / 6), 256 * sizeof(double));
    cudaMemset(d_omega_y_cos, cos(M_PI / 6), 256 * sizeof(double));
    check_cuda_errors();

    allowed_steps = 256;

    // compute number of blocks and threads
    n_samples = x.size();
    n_threads = NT;
    n_blocks = (n_samples + n_threads - 1) / n_threads;

    // initialize curandom states
    cudaMalloc(&d_rand_states, n_threads * n_blocks * sizeof(curandState));
    setup_random_states<<<n_blocks, n_threads>>>(d_rand_states, clock());
    check_cuda_errors();

    std::cout << "Done initializing GPU henon..." << std::endl;
}

gpu_henon::~gpu_henon()
{
    cudaFree(d_x);
    cudaFree(d_px);
    cudaFree(d_y);
    cudaFree(d_py);
    cudaFree(d_steps);
    cudaFree(d_rand_states);
    cudaFree(d_omega_x_sin);
    cudaFree(d_omega_x_cos);
    cudaFree(d_omega_y_sin);
    cudaFree(d_omega_y_cos);
    check_cuda_errors();
}

void gpu_henon::compute_a_modulation(unsigned int n_turns, double omega_x, double omega_y, std::string modulation_kind, double omega_0, double epsilon, unsigned int offset)
{
    std::cout << "Computing a modulation..." << std::endl;
    // compute a modulation
    tie(omega_x_vec, omega_y_vec) = pick_a_modulation(n_turns, omega_x, omega_y, modulation_kind, omega_0, epsilon, offset);

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

    // free old modulations
    cudaFree(d_omega_x_sin);
    cudaFree(d_omega_x_cos);
    cudaFree(d_omega_y_sin);
    cudaFree(d_omega_y_cos);

    check_cuda_errors();

    // copy to gpu
    cudaMalloc(&d_omega_x_sin, omega_x_sin.size() * sizeof(double));
    cudaMalloc(&d_omega_x_cos, omega_x_cos.size() * sizeof(double));
    cudaMalloc(&d_omega_y_sin, omega_y_sin.size() * sizeof(double));
    cudaMalloc(&d_omega_y_cos, omega_y_cos.size() * sizeof(double));

    check_cuda_errors();

    cudaMemcpy(d_omega_x_sin, omega_x_sin.data(), omega_x_sin.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_omega_x_cos, omega_x_cos.data(), omega_x_cos.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_omega_y_sin, omega_y_sin.data(), omega_y_sin.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_omega_y_cos, omega_y_cos.data(), omega_y_cos.size() * sizeof(double), cudaMemcpyHostToDevice);

    check_cuda_errors();
    allowed_steps = n_turns;
    std::cout << "Done computing a modulation..." << std::endl;
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

void gpu_henon::track(unsigned int n_turns, double mu, double barrier, double kick_module, bool inverse)
{
    // check if n_turns is valid
    if (inverse)
    {
        if (n_turns > global_steps)
            throw std::runtime_error("The number of turns is too large.");
    }
    else
    {
        if (n_turns + global_steps > allowed_steps)
            throw std::runtime_error("The number of turns is too large.");
    }
    
    for (unsigned int j = 0; j < n_turns; j++)
    {
        gpu_henon_track<<<n_blocks, n_threads>>>(
            d_x, d_px, d_y, d_py, d_steps,
            n_samples, 1, barrier * barrier, mu,
            d_omega_x_sin, d_omega_x_cos, d_omega_y_sin, d_omega_y_cos,
            kick_module, d_rand_states, inverse);
        
        if (j % 1000 == 0)
            check_cuda_errors();
    }   
    // Update the counter
    if (!inverse)
        global_steps += n_turns;
    else
        global_steps -= n_turns;
}

std::vector<std::vector<double>> gpu_henon::track_MEGNO(std::vector<unsigned int> n_turns, double mu, double barrier, double kick_module, bool inverse)
{
    // pre allocate megno space and fill it with NaNs
    std::vector<std::vector<double>> megno(n_turns.size(), std::vector<double>(n_samples / 2, std::numeric_limits<double>::quiet_NaN()));
    unsigned int index = 0;

    double *d_displacement_1;
    double *d_displacement_2;
    double *d_ratio_sum;

    cudaMalloc(&d_displacement_1, (n_samples / 2) * sizeof(double));
    cudaMalloc(&d_displacement_2, (n_samples / 2) * sizeof(double));
    cudaMalloc(&d_ratio_sum, (n_samples / 2) * sizeof(double));

    // initialize vectors to 0
    cudaMemset(d_displacement_1, 0, (n_samples / 2) * sizeof(double));
    cudaMemset(d_displacement_2, 0, (n_samples / 2) * sizeof(double));
    cudaMemset(d_ratio_sum, 0, (n_samples / 2) * sizeof(double));

    gpu_compute_displacements<<<n_blocks, n_threads>>>(
        d_x, d_px, d_y, d_py, d_displacement_2, n_samples / 2);
    // run the simulation
    for (unsigned int j = 0; j < n_turns.back(); j++)
    {
        gpu_henon_track<<<n_blocks, n_threads>>>(
            d_x, d_px, d_y, d_py, d_steps,
            n_samples, 1, barrier * barrier, mu,
            d_omega_x_sin, d_omega_x_cos, d_omega_y_sin, d_omega_y_cos,
            kick_module, d_rand_states, inverse);

        gpu_compute_displacements<<<n_blocks, n_threads>>>(
            d_x, d_px, d_y, d_py,
            j%2 == 0 ? d_displacement_1 : d_displacement_2,
            n_samples / 2
        );
        
        if (j % 2 == 0)
            gpu_add_to_ratio<<<n_blocks, n_threads>>>(
                d_displacement_1, d_displacement_2,
                d_ratio_sum, j + 1, n_samples / 2
            );
        else
            gpu_add_to_ratio<<<n_blocks, n_threads>>>(
                d_displacement_2, d_displacement_1,
                d_ratio_sum, j + 1, n_samples / 2
            );
            
        if (j % 1000 == 0)
            check_cuda_errors();

        // if j is in the n_turns vector, then we need to compute megno
        if (j + 1 == n_turns[index])
        {
            cudaMemcpy(
                megno[index].data(), d_ratio_sum, (n_samples / 2) * sizeof(double), cudaMemcpyDeviceToHost);

            // divide by number of turns
            for (size_t i = 0; i < megno[index].size(); i++)
            {
                megno[index][i] /= n_turns[index] * 0.5;
            }
            index += 1;
        }
    }

    // clear the displacement vectors
    cudaFree(d_displacement_1);
    cudaFree(d_displacement_2);
    cudaFree(d_ratio_sum);

    // Update the counter
    if (!inverse)
        global_steps += n_turns.back();
    else
        global_steps -= n_turns.back();

    return megno;
}

std::vector<std::vector<double>> gpu_henon::track_realignments(std::vector<unsigned int> n_turns, double mu, double barrier, double kick_module, bool inverse, double low_module, double barrier_module)
{
    // pre allocate megno space and fill it with NaNs
    std::vector<std::vector<double>> disp_values(n_turns.size(), std::vector<double>(n_samples / 2, std::numeric_limits<double>::quiet_NaN()));
    
    std::vector<double> disp_tmp(n_samples / 2);
    unsigned int index = 0;

    double *d_displacement_realign;
    double *d_displacement_at_check;

    cudaMalloc(&d_displacement_realign, (n_samples / 2) * sizeof(double));
    cudaMalloc(&d_displacement_at_check, (n_samples / 2) * sizeof(double));

    // initialize vectors to 0
    cudaMemset(d_displacement_realign, 0, (n_samples / 2) * sizeof(double));
    cudaMemset(d_displacement_at_check, 0, (n_samples / 2) * sizeof(double));

    // run the simulation
    for (unsigned int j = 0; j < n_turns.back(); j++)
    {
        gpu_henon_track<<<n_blocks, n_threads>>>(
            d_x, d_px, d_y, d_py, d_steps,
            n_samples, 1, barrier * barrier, mu,
            d_omega_x_sin, d_omega_x_cos, d_omega_y_sin, d_omega_y_cos,
            kick_module, d_rand_states, inverse);

        gpu_compute_displacements_and_realign<<<n_blocks, n_threads>>>(d_x, d_px, d_y, d_py, d_displacement_realign, n_samples / 2, low_module, barrier_module);

        if (j % 1000 == 0)
            check_cuda_errors();

        if (j + 1 == n_turns[index])
        {
            gpu_compute_displacements<<<n_blocks, n_threads>>>(d_x, d_px, d_y, d_py, d_displacement_at_check, n_samples / 2);
            gpu_sum_two_arrays<<<n_blocks, n_threads>>>(d_displacement_at_check, d_displacement_realign, n_samples / 2);
            cudaMemcpy(
                disp_values[index].data(), d_displacement_at_check, (n_samples / 2) * sizeof(double), cudaMemcpyDeviceToHost);
            index += 1;
            std::cout << "(" << n_turns[index - 1] << "/" << n_turns.back() << ")" << std::endl;
        }
    }

    // clear the displacement vectors
    cudaFree(d_displacement_realign);
    cudaFree(d_displacement_at_check);

    // Update the counter
    if (!inverse)
        global_steps += n_turns.back();
    else
        global_steps -= n_turns.back();

    return disp_values;
}

// getters

std::vector<double> gpu_henon::get_x() const
{ 
    std::vector<double> x_out(n_samples);
    cudaMemcpy(x_out.data(), d_x, n_samples * sizeof(double), cudaMemcpyDeviceToHost);
    return x_out;
}
std::vector<double> gpu_henon::get_px() const
{ 
    std::vector<double> px_out(n_samples);
    cudaMemcpy(px_out.data(), d_px, n_samples * sizeof(double), cudaMemcpyDeviceToHost);
    return px_out;
}
std::vector<double> gpu_henon::get_y() const 
{
    std::vector<double> y_out(n_samples);
    cudaMemcpy(y_out.data(), d_y, n_samples * sizeof(double), cudaMemcpyDeviceToHost);
    return y_out;
}
std::vector<double> gpu_henon::get_py() const
{
    std::vector<double> py_out(n_samples);
    cudaMemcpy(py_out.data(), d_py, n_samples * sizeof(double), cudaMemcpyDeviceToHost);
    return py_out;
}
std::vector<unsigned int> gpu_henon::get_steps() const
{
    std::vector<unsigned int> steps_out(n_samples);
    cudaMemcpy(steps_out.data(), d_steps, n_samples * sizeof(int), cudaMemcpyDeviceToHost);
    return steps_out;
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
                     const std::vector<double> &py_init) : 
    henon(x_init, px_init, y_init, py_init)
{}

cpu_henon::~cpu_henon() {}

void cpu_henon::track(unsigned int n_turns, double mu, double barrier, double kick_module, bool inverse)
{
    // check if n_turns is valid
    if (inverse)
    {
        if (n_turns > global_steps)
            throw std::runtime_error("The number of turns is too large.");
    }
    else
    {
        if (n_turns + global_steps > allowed_steps)
            throw std::runtime_error("The number of turns is too large.");
    }
    
    double barrier_pow_2 = barrier * barrier;

    // for every element in vector x, execute cpu_henon_track in parallel
    std::vector<std::thread> threads;
    for (unsigned int i = 0; i < n_threads_cpu; i++)
    {
        threads.push_back(std::thread(
            [&](int thread_idx)
            {
                std::mt19937_64 rng;
                for (unsigned int j = thread_idx; j < x.size(); j+=n_threads_cpu)
                {
                    for (unsigned int k = 0; k < n_turns; k++)
                    {
                        henon_step(
                            x[j], px[j], y[j], py[j], steps[j],
                            omega_x_sin.data(), omega_x_cos.data(),
                            omega_y_sin.data(), omega_y_cos.data(),
                            barrier_pow_2, mu, kick_module,
                            rng, inverse);
                    }
                }
            },
            i));
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

std::vector<std::vector<double>> cpu_henon::track_MEGNO(
    std::vector<unsigned int> n_turns, double mu, double barrier, double kick_module, bool inverse)
{
    // pre allocate megno space and fill it with NaNs
    std::vector<std::vector<double>> megno(n_turns.size(), std::vector<double>(x.size() / 2, std::numeric_limits<double>::quiet_NaN()));

    std::vector<double> displacement_1(x.size() / 2, 0.0);
    std::vector<double> displacement_2(x.size() / 2, 0.0);
    std::vector<double> ratio_sum(x.size() / 2, 0.0);

    // check if n_turns is valid
    if (inverse)
    {
        if (n_turns.back() > global_steps)
            throw std::runtime_error("The number of turns is too large.");
    }
    else
    {
        if (n_turns.back() + global_steps > allowed_steps)
            throw std::runtime_error("The number of turns is too large.");
    }

    double barrier_pow_2 = barrier * barrier;

    // for every element in vector x, execute cpu_henon_track in parallel
    std::vector<std::thread> threads;
    for (unsigned int i = 0; i < n_threads_cpu; i++)
    {
        threads.push_back(std::thread(
            [&](int thread_idx)
            {
                std::mt19937_64 rng;
                for (unsigned int j = thread_idx; j < x.size() / 2; j += n_threads_cpu)
                {
                    unsigned int index = 0;
                    displacement_2[j] = displacement(
                        x[j], px[j], y[j], py[j],
                        x[j + x.size() / 2], px[j + x.size() / 2],
                        y[j + x.size() / 2], py[j + x.size() / 2]);
                    for (unsigned int k = 0; k < n_turns.back(); k++)
                    {
                        henon_step(
                            x[j], px[j], y[j], py[j], steps[j],
                            omega_x_sin.data(), omega_x_cos.data(),
                            omega_y_sin.data(), omega_y_cos.data(),
                            barrier_pow_2, mu, kick_module,
                            rng, inverse);
                        henon_step(
                            x[j + x.size() / 2], px[j + x.size() / 2],
                            y[j + x.size() / 2], py[j + x.size() / 2],
                            steps[j + x.size() / 2],
                            omega_x_sin.data(), omega_x_cos.data(),
                            omega_y_sin.data(), omega_y_cos.data(),
                            barrier_pow_2, mu, kick_module,
                            rng, inverse);

                        if (k % 2 == 0)
                        {
                            displacement_1[j] = displacement(
                                x[j], px[j], y[j], py[j],
                                x[j + x.size() / 2], px[j + x.size() / 2],
                                y[j + x.size() / 2], py[j + x.size() / 2]);
                        }
                        else
                        {
                            displacement_2[j] = displacement(
                                x[j], px[j], y[j], py[j],
                                x[j + x.size() / 2], px[j + x.size() / 2],
                                y[j + x.size() / 2], py[j + x.size() / 2]);
                        }

                        if (k % 1000 == 0)
                            check_keyboard_interrupt();

                        if (k % 2 == 0)
                        {
                            ratio_sum[j] += k * log10(displacement_1[j] / displacement_2[j]);
                        }
                        else
                        {
                            ratio_sum[j] += k * log10(displacement_2[j] / displacement_1[j]);
                        }

                        if (k + 1 == n_turns[index])
                        {
                            megno[index][j] = 2.0 * ratio_sum[j] / (n_turns[index]);
                            index += 1;
                        }
                    }
                }
            },
            i));
    }

    // join threads
    for (auto &t : threads)
        t.join();

    // update global_steps
    if (!inverse)
        global_steps += n_turns.back();
    else
        global_steps -= n_turns.back();

    return megno;
}

std::vector<std::vector<double>> cpu_henon::track_realignments(
    std::vector<unsigned int> n_turns, double mu, double barrier, double kick_module, bool inverse, double low_module, double barrier_module)
{
    // pre allocate displacement space and fill it with NaNs
    std::vector<std::vector<double>> disp_values(n_turns.size(), std::vector<double>(x.size() / 2, std::numeric_limits<double>::quiet_NaN()));

    std::vector<double> displacement_realign(x.size() / 2, 0.0);

    // check if n_turns is valid
    if (inverse)
    {
        if (n_turns.back() > global_steps)
            throw std::runtime_error("The number of turns is too large.");
    }
    else
    {
        if (n_turns.back() + global_steps > allowed_steps)
            throw std::runtime_error("The number of turns is too large.");
    }

    double barrier_pow_2 = barrier * barrier;

    // for every element in vector x, execute cpu_henon_track in parallel
    std::vector<std::thread> threads;
    for (unsigned int i = 0; i < n_threads_cpu; i++)
    {
        threads.push_back(std::thread(
            [&](int thread_idx)
            {
                std::mt19937_64 rng;
                for (unsigned int j = thread_idx; j < x.size() / 2; j += n_threads_cpu)
                {
                    unsigned int index = 0;
                    for (unsigned int k = 0; k < n_turns.back(); k++)
                    {
                        henon_step(
                            x[j], px[j], y[j], py[j], steps[j],
                            omega_x_sin.data(), omega_x_cos.data(),
                            omega_y_sin.data(), omega_y_cos.data(),
                            barrier_pow_2, mu, kick_module,
                            rng, inverse);
                        henon_step(
                            x[j + x.size() / 2], px[j + x.size() / 2],
                            y[j + x.size() / 2], py[j + x.size() / 2],
                            steps[j + x.size() / 2],
                            omega_x_sin.data(), omega_x_cos.data(),
                            omega_y_sin.data(), omega_y_cos.data(),
                            barrier_pow_2, mu, kick_module,
                            rng, inverse);

                        if (check_nan(x[j], px[j], y[j], py[j]) || check_nan(x[j + x.size() / 2], px[j + x.size() / 2], y[j + x.size() / 2], py[j + x.size() / 2]))
                        {
                            break;
                        }

                        double tmp_displacement = displacement(
                            x[j], px[j], y[j], py[j],
                            x[j + x.size() / 2], px[j + x.size() / 2],
                            y[j + x.size() / 2], py[j + x.size() / 2]);
                        
                        if (tmp_displacement > barrier_module)
                        {
                            displacement_realign[j] += tmp_displacement - low_module;
                            realign(
                                x[j + x.size() / 2], px[j + x.size() / 2],
                                y[j + x.size() / 2], py[j + x.size() / 2],
                                x[j], px[j], y[j], py[j],
                                tmp_displacement, low_module
                            );
                            tmp_displacement = low_module;
                        }

                        if (k % 1000 == 0)
                            check_keyboard_interrupt();

                        if (k + 1 == n_turns[index])
                        {
                            disp_values[index][j] = tmp_displacement + displacement_realign[j];
                            index += 1;
                        }
                    }
                }
            },
            i));
    }

    // join threads
    for (auto &t : threads)
        t.join();

    // update global_steps
    if (!inverse)
        global_steps += n_turns.back();
    else
        global_steps -= n_turns.back();

    return disp_values;
}

std::vector<std::vector<double>> henon::track_tangent_map(
    std::vector<unsigned int> n_turns, double mu, double barrier, double kick_module, bool inverse)
{
    // pre allocate tangent_map space and fill it with NaNs
    std::vector<std::vector<double>> tm(n_turns.size(), std::vector<double>(x.size(), std::numeric_limits<double>::quiet_NaN()));

    // check if n_turns is valid
    if (inverse)
    {
        if (n_turns.back() > global_steps)
            throw std::runtime_error("The number of turns is too large.");
    }
    else
    {
        if (n_turns.back() + global_steps > allowed_steps)
            throw std::runtime_error("The number of turns is too large.");
    }

    double barrier_pow_2 = barrier * barrier;

    // for every element in vector x, execute cpu_henon_track in parallel
    std::vector<std::thread> threads;
    for (unsigned int i = 0; i < n_threads_cpu; i++)
    {
        threads.push_back(std::thread(
            [&](int thread_idx)
            {
                std::mt19937_64 rng;
                for (unsigned int j = thread_idx; j < x.size(); j += n_threads_cpu)
                {
                    unsigned int index = 0;

                    std::array<std::array<double, 4>, 4> tangent_map;
                    // initialize tangent_map as unit matrix
                    for (unsigned int k = 0; k < 4; k++)
                    {
                        for (unsigned int l = 0; l < 4; l++)
                        {
                            if (k == l)
                                tangent_map[k][l] = 1.0;
                            else
                                tangent_map[k][l] = 0.0;
                        }
                    }
                    std::array<std::array<double, 4>, 4> tangent_map_temp;
                    std::array<std::array<double, 4>, 4> matrix_mult_temp;

                    for (unsigned int k = 0; k < n_turns.back(); k++)
                    {
                        henon_step(
                            x[j], px[j], y[j], py[j], steps[j],
                            omega_x_sin.data(), omega_x_cos.data(),
                            omega_y_sin.data(), omega_y_cos.data(),
                            barrier_pow_2, mu, kick_module,
                            rng, inverse);

                        if (check_nan(x[j], px[j], y[j], py[j]))
                        {
                            break;
                        }

                        tangent_map_temp = tangent_matrix(
                            x[j], px[j], y[j], py[j],
                            omega_x_sin[steps[j]], omega_x_cos[steps[j]],
                            omega_y_sin[steps[j]], omega_y_cos[steps[j]],
                            mu);

                        // multiply tangent_map_temp with tangent_map
                        for (unsigned int idx1 = 0; idx1 < 4; idx1++)
                        {
                            for (unsigned int idx2 = 0; idx2 < 4; idx2++)
                            {
                                matrix_mult_temp[idx1][idx2] = 0.0;
                                for (unsigned int idx3 = 0; idx3 < 4; idx3++)
                                {
                                    matrix_mult_temp[idx1][idx2] += tangent_map_temp[idx1][idx3] * tangent_map[idx3][idx2];
                                }
                            }
                        }
                        // copy matrix_mult_temp to tangent_map
                        for (unsigned int idx1 = 0; idx1 < 4; idx1++)
                        {
                            for (unsigned int idx2 = 0; idx2 < 4; idx2++)
                            {
                                tangent_map[idx1][idx2] = matrix_mult_temp[idx1][idx2];
                            }
                        }

                        if (k % 1000 == 0)
                            check_keyboard_interrupt();

                        if (k + 1 == n_turns[index])
                        {
                            double indicator = 0.0;
                            for (unsigned int idx1 = 0; idx1 < 4; idx1++)
                            {
                                for (unsigned int idx2 = 0; idx2 < 4; idx2++)
                                {
                                    indicator += tangent_map[idx1][idx2] * tangent_map[idx1][idx2];
                                }
                            }
                            tm[index][j] = indicator;
                            index += 1;
                        }
                    }
                }
            },
            i));
    }

    // join threads
    for (auto &t : threads)
        t.join();

    // update global_steps
    if (!inverse)
        global_steps += n_turns.back();
    else
        global_steps -= n_turns.back();

    return tm;
}