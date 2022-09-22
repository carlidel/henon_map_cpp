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

__device__ void tangent_matrix(const double &x, const double &px, const double &y, const double &py, const double &sx, const double &cx, const double &sy, const double &cy, const double &mu, double *out, size_t idx)
{
    out[idx + 0 * 4 + 0] = cx + sx * (2 * x + mu * 3 * x * x - mu * 3 * y * y);
    out[idx + 0 * 4 + 1] = sx;
    out[idx + 0 * 4 + 2] = sx * (-2 * y - mu * 6 * x * y);
    out[idx + 0 * 4 + 3] = 0.0;

    out[idx + 1 * 4 + 0] = -sx + cx * (2 * x + mu * 3 * x * x - mu * 3 * y * y);
    out[idx + 1 * 4 + 1] = cx;
    out[idx + 1 * 4 + 2] = cx * (-2 * y - mu * 6 * x * y);
    out[idx + 1 * 4 + 3] = 0.0;

    out[idx + 2 * 4 + 0] = sy * (-2 * y - mu * 6 * x * y);
    out[idx + 2 * 4 + 1] = 0.0;
    out[idx + 2 * 4 + 2] = cy + sy * (-2 * x - mu * 3 * x * x + mu * 3 * y * y);
    out[idx + 2 * 4 + 3] = sy;

    out[idx + 3 * 4 + 0] = cy * (-2 * y - mu * 6 * x * y);
    out[idx + 3 * 4 + 1] = 0.0;
    out[idx + 3 * 4 + 2] = -sy + cy * (-2 * x - mu * 3 * x * x + mu * 3 * y * y);
    out[idx + 3 * 4 + 3] = cy;
}

__host__ std::vector<std::vector<double>> tangent_matrix(const double &x, const double &px, const double &y, const double &py, const double &sx, const double &cx, const double &sy, const double &cy, const double &mu)
{
    std::vector<std::vector<double>> matrix(4, std::vector<double>(4, 0.0));
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

__host__ std::vector<std::vector<double>> inverse_tangent_matrix(const double &x, const double &px, const double &y, const double &py, const double &sx, const double &cx, const double &sy, const double &cy, const double &mu)
{
    std::vector<std::vector<double>> matrix(4, std::vector<double>(4, 0.0));

    matrix[0][0] = cx;
    matrix[0][1] = -sx;
    matrix[0][2] = 0;
    matrix[0][3] = 0;

    matrix[1][0] = -3 * cx * mu * (pow(cx * x - px * sx, 2) - pow(cy * y - py * sy, 2)) - 2 * cx * (cx * x - px * sx) + sx;
    matrix[1][1] = cx + 3 * mu * sx * (pow(cx * x - px * sx, 2) - pow(cy * y - py * sy, 2)) + 2 * sx * (cx * x - px * sx);
    matrix[1][2] = 2 * cy * (cy * y - py * sy) * (3 * mu * (cx * x - px * sx) + 1);
    matrix[1][3] = 2 * sy * (cy * y - py * sy) * (-3 * mu * (cx * x - px * sx) - 1);

    matrix[2][0] = 0;
    matrix[2][1] = 0;
    matrix[2][2] = cy;
    matrix[2][3] = -sy;

    matrix[3][0] = 2 * cx * (cy * y - py * sy) * (3 * mu * (cx * x - px * sx) + 1);
    matrix[3][1] = 2 * sx * (cy * y - py * sy) * (-3 * mu * (cx * x - px * sx) - 1);
    matrix[3][2] = 3 * cy * mu * (pow(cx * x - px * sx, 2) - pow(cy * y - py * sy, 2)) + 2 * cy * (cx * x - px * sx) + sy;
    matrix[3][3] = cy - 3 * mu * sy * (pow(cx * x - px * sx, 2) - pow(cy * y - py * sy, 2)) - 2 * sy * (cx * x - px * sx);

    return matrix;
}

std::vector<std::vector<double>> multiply_matrices(const std::vector<std::vector<double>> &matrix1, const std::vector<std::vector<double>> &matrix2)
{
    std::vector<std::vector<double>> matrix(4, std::vector<double>(4, 0.0));
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            for (int k = 0; k < 4; k++)
            {
                matrix[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
    return matrix;
}

__global__ void get_matrices_and_multiply(double *matrices, const double *x, const double *px, const double *y, const double *py, const double *sx, const double *cx, const double *sy, const double *cy, const double mu, const unsigned int global_steps, const int size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size)
        return;
    
    double tmp_matrix[16];
    double result_matrix[16];

    tangent_matrix(x[idx], px[idx], y[idx], py[idx], sx[global_steps], cx[global_steps], sy[global_steps], cy[global_steps], mu, tmp_matrix, 0);

    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            result_matrix[i * 4 + j] = 0.0;
            for (int k = 0; k < 4; k++)
            {
                result_matrix[i * 4 + j] += tmp_matrix[i * 4 + k] * matrices[idx * 16 + k * 4 + j];
            }
        }
    }
    // copy result matrix to global memory
    for (int i = 0; i < 16; i++)
    {
        matrices[idx * 16 + i] = result_matrix[i];
    }
}

__global__ void get_matrices_and_set(double *matrices, const double *x, const double *px, const double *y, const double *py, const double *sx, const double *cx, const double *sy, const double *cy, const double mu, const unsigned int global_steps, const int size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size)
        return;
    
    double tmp_matrix[16];

    tangent_matrix(x[idx], px[idx], y[idx], py[idx], sx[global_steps], cx[global_steps], sy[global_steps], cy[global_steps], mu, tmp_matrix, 0);

    // copy result matrix to global memory
    for (int i = 0; i < 16; i++)
    {
        matrices[idx * 16 + i] = tmp_matrix[i];
    }
}


__global__ void matrix_vector_multiply(const double *matrices, double *vectors, const size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size)
        return;

    double tmp_vector[4];

    for (int i = 0; i < 4; i++)
    {
        tmp_vector[i] = 0.0;
        for (int j = 0; j < 4; j++)
        {
            tmp_vector[i] += matrices[idx * 16 + i * 4 + j] * vectors[idx * 4 + j];
        }
    }

    for (int i = 0; i < 4; i++)
    {
        vectors[idx * 4 + i] = tmp_vector[i];
    }
}


__global__ void normalize_vectors(double *vectors, const size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size)
        return;

    double norm = 0.0;
    for (int i = 0; i < 4; i++)
    {
        norm += vectors[idx * 4 + i] * vectors[idx * 4 + i];
    }
    norm = sqrt(norm);

    for (int i = 0; i < 4; i++)
    {
        vectors[idx * 4 + i] /= norm;
    }
}


__host__ void random_4d_kick(double &x, double &px, double &y, double &py, const double &kick_module, std::mt19937_64 &generator)
{
    // create normal distribution with mean 0 and standard deviation 1
    auto normal_dist = std::normal_distribution<double>(0.0, 1.0);
    // create the array
    std::array<double, 4> kick;
    for (auto &kick_component : kick)
    {
        kick_component = normal_dist(generator);
    }
    // compute the module of the kick
    double mod = sqrt(
        kick[0] * kick[0] +
        kick[1] * kick[1] +
        kick[2] * kick[2] +
        kick[3] * kick[3]
    );
    
    // rescale the kick
    x  += kick[0] * kick_module / mod;
    px += kick[1] * kick_module / mod;
    y  += kick[2] * kick_module / mod;
    py += kick[3] * kick_module / mod;
}

__device__ void random_4d_kick(double &x, double &px, double &y, double &py, const double &kick_module, curandState &state)
{
    double rand_kicks[4];
    rand_kicks[0] = curand_normal(&state);
    rand_kicks[1] = curand_normal(&state);
    rand_kicks[2] = curand_normal(&state);
    rand_kicks[3] = curand_normal(&state);
    
    double mod = sqrt(
        rand_kicks[0] * rand_kicks[0] +
        rand_kicks[1] * rand_kicks[1] +
        rand_kicks[2] * rand_kicks[2] +
        rand_kicks[3] * rand_kicks[3]
    );
    x  += rand_kicks[0] * kick_module / mod;
    px += rand_kicks[1] * kick_module / mod;
    y  += rand_kicks[2] * kick_module / mod;
    py += rand_kicks[3] * kick_module / mod;
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
    double *out, const size_t size, const double low_value=1, const bool log_me=false)
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

    if (log_me)
    {
        out[i] = log(displacement(x1[i], px1[i], y1[i], py1[i], x2, px2, y2, py2) / low_value);
    }
    else
    {
        out[i] = displacement(x1[i], px1[i], y1[i], py1[i], x2, px2, y2, py2);
    }
}

__global__ void gpu_compute_displacements_and_realign(
    double *x1, double *px1, double *y1, double *py1,
    double *out, const size_t size, const double low_module)
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
    out[i] += log(tmp_displacement/low_module);
    realign(
        x1[i + size], px1[i + size], y1[i + size], py1[i + size],
        x1[i], px1[i], y1[i], py1[i], tmp_displacement, low_module);

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

particles_4d::particles_4d(const std::vector<double> &x_,
                           const std::vector<double> &px_,
                           const std::vector<double> &y_,
                           const std::vector<double> &py_)
    : x(x_), px(px_), y(y_), py(py_)
{
    x0 = x;
    px0 = px;
    y0 = y;
    py0 = py;
    steps.resize(x.size(), 0);
    valid.resize(x.size(), true);
    ghost.resize(x.size(), false);

    idx.resize(x.size());
    idx_base.resize(x.size());
    for (size_t i = 0; i < x.size(); i++)
    {
        idx[i] = i;
        idx_base[i] = i;
    }

    n_particles = x.size();
    n_ghosts_per_particle = 0;

    global_steps = 0;
}

void particles_4d::reset()
{
    x = x0;
    px = px0;
    y = y0;
    py = py0;
    steps = std::vector<unsigned int>(x.size(), 0);
    valid = std::vector<uint8_t>(x.size(), true);
}

void particles_4d::add_ghost(const double &displacement_module, const std::string &direction)
{
    std::vector<double> x_ghost;
    std::vector<double> px_ghost;
    std::vector<double> y_ghost;
    std::vector<double> py_ghost;
    std::vector<unsigned int> steps_ghost;
    std::vector<uint8_t> valid_ghost;
    std::vector<uint8_t> ghost_ghost;
    std::vector<size_t> idx_ghost;
    std::vector<size_t> idx_base_ghost;

    for (size_t i = 0; i < x.size(); i++)
    {
        if (!ghost[i])
        {
            steps_ghost.push_back(steps[i]);
            valid_ghost.push_back(valid[i]);
            ghost_ghost.push_back(true);
            idx_ghost.push_back(n_ghosts_per_particle);
            idx_base_ghost.push_back(i);

            if (direction == "x")
            {
                x_ghost.push_back(x[i] + displacement_module);
                px_ghost.push_back(px[i]);
                y_ghost.push_back(y[i]);
                py_ghost.push_back(py[i]);
            }
            else if (direction == "y")
            {
                x_ghost.push_back(x[i]);
                px_ghost.push_back(px[i]);
                y_ghost.push_back(y[i] + displacement_module);
                py_ghost.push_back(py[i]);
            }
            else if (direction == "px")
            {
                x_ghost.push_back(x[i]);
                px_ghost.push_back(px[i] + displacement_module);
                y_ghost.push_back(y[i]);
                py_ghost.push_back(py[i]);
            }
            else if (direction == "py")
            {
                x_ghost.push_back(x[i]);
                px_ghost.push_back(px[i]);
                y_ghost.push_back(y[i]);
                py_ghost.push_back(py[i] + displacement_module);
            }
            else if (direction == "random")
            {
                x_ghost.push_back(x[i]);
                px_ghost.push_back(px[i]);
                y_ghost.push_back(y[i]);
                py_ghost.push_back(py[i]);
                random_4d_kick(x_ghost.back(), px_ghost.back(), y_ghost.back(), py_ghost.back(), displacement_module, rng);
            }
            else
            {
                throw std::runtime_error("Invalid direction.");
            }
        }
    }

    // concatenate ghost particles
    x.insert(x.end(), x_ghost.begin(), x_ghost.end());
    px.insert(px.end(), px_ghost.begin(), px_ghost.end());
    y.insert(y.end(), y_ghost.begin(), y_ghost.end());
    py.insert(py.end(), py_ghost.begin(), py_ghost.end());
    x0.insert(x0.end(), x_ghost.begin(), x_ghost.end());
    px0.insert(px0.end(), px_ghost.begin(), px_ghost.end());
    y0.insert(y0.end(), y_ghost.begin(), y_ghost.end());
    py0.insert(py0.end(), py_ghost.begin(), py_ghost.end());
    steps.insert(steps.end(), steps_ghost.begin(), steps_ghost.end());
    valid.insert(valid.end(), valid_ghost.begin(), valid_ghost.end());
    ghost.insert(ghost.end(), ghost_ghost.begin(), ghost_ghost.end());
    idx.insert(idx.end(), idx_ghost.begin(), idx_ghost.end());
    idx_base.insert(idx_base.end(), idx_base_ghost.begin(), idx_base_ghost.end());

    n_ghosts_per_particle += 1;
}

void particles_4d::renormalize(const double &module_target)
{
    // get number of cpu threads
    const unsigned int n_threads = std::thread::hardware_concurrency();

    std::vector<std::thread> threads;

    // create threads
    for (unsigned int t = 0; t < n_threads; t++)
    {
        threads.push_back(std::thread(
            [&](unsigned int thread_idx)
            {
                for (size_t i = thread_idx; i < x.size(); i += n_threads)
                {
                    if(!ghost[i])
                        continue;

                    size_t base_idx = idx_base[i];
                    double module = displacement(
                        x[i], px[i], y[i], py[i],
                        x[base_idx], px[base_idx], y[base_idx], py[base_idx]);
                    
                    realign(x[i], px[i], y[i], py[i],
                            x[base_idx], px[base_idx], y[base_idx], py[base_idx],
                            module, module_target);
                }
            },
            t
        ));
    }

    // join threads
    for (auto &thread : threads)
    {
        thread.join();
    }
}

const std::vector<std::vector<double>> particles_4d::get_displacement_module() const
{
    std::vector<std::vector<double>> val_displacement(n_particles, std::vector<double>(n_ghosts_per_particle, std::numeric_limits<double>::quiet_NaN()));

    // get number of cpu threads
    const unsigned int n_threads = std::thread::hardware_concurrency();

    std::vector<std::thread> threads;

    // create threads
    for (unsigned int t = 0; t < n_threads; t++)
    {
        threads.push_back(std::thread(
            [&](unsigned int thread_idx)
            {
                for (size_t i = thread_idx; i < x.size(); i += n_threads)
                {
                    if (!ghost[i])
                        continue;

                    size_t base_idx = idx_base[i];
                    double module = displacement(
                        x[i], px[i], y[i], py[i],
                        x[base_idx], px[base_idx], y[base_idx], py[base_idx]);
                    val_displacement[base_idx][idx[i]] = module;
                }
            },
            t));
    }

    // join threads
    for (auto &thread : threads)
    {
        thread.join();
    }

    return val_displacement;
}

const std::vector<std::vector<std::vector<double>>> particles_4d::get_displacement_direction() const
{
    std::vector<std::vector<std::vector<double>>> dir_displacement(
        {{
            std::vector<std::vector<double>>(n_particles, std::vector<double>(n_ghosts_per_particle, std::numeric_limits<double>::quiet_NaN())),
            std::vector<std::vector<double>>(n_particles, std::vector<double>(n_ghosts_per_particle, std::numeric_limits<double>::quiet_NaN())),
            std::vector<std::vector<double>>(n_particles, std::vector<double>(n_ghosts_per_particle, std::numeric_limits<double>::quiet_NaN())),
            std::vector<std::vector<double>>(n_particles, std::vector<double>(n_ghosts_per_particle, std::numeric_limits<double>::quiet_NaN()))
    }});

    // get number of cpu threads
    const unsigned int n_threads = std::thread::hardware_concurrency();

    std::vector<std::thread> threads;

    // create threads
    for (unsigned int t = 0; t < n_threads; t++)
    {
        threads.push_back(std::thread(
            [&](unsigned int thread_idx)
            {
                for (size_t i = thread_idx; i < x.size(); i += n_threads)
                {
                    if (!ghost[i])
                        continue;

                    size_t base_idx = idx_base[i];
                    double module = displacement(
                        x[i], px[i], y[i], py[i],
                        x[base_idx], px[base_idx], y[base_idx], py[base_idx]);
                    
                    std::array<double, 4> dir = {x[i], px[i], y[i], py[i]};

                    realign(dir[0], dir[1], dir[2], dir[3],
                            x[base_idx], px[base_idx], y[base_idx], py[base_idx],
                            module, 1.0);
                    dir_displacement[0][base_idx][idx[i]] = dir[0] - x[base_idx];
                    dir_displacement[1][base_idx][idx[i]] = dir[1] - px[base_idx];
                    dir_displacement[2][base_idx][idx[i]] = dir[2] - y[base_idx];
                    dir_displacement[3][base_idx][idx[i]] = dir[3] - py[base_idx];
                }
            },
            t
        ));

    }

    // join threads
    for (auto &thread : threads)
    {
        thread.join();
    }

    return dir_displacement;
}

const std::vector<double> particles_4d::get_x() const
{
    return x;
}

const std::vector<double> particles_4d::get_px() const
{
    return px;
}

const std::vector<double> particles_4d::get_y() const
{
    return y;
}

const std::vector<double> particles_4d::get_py() const
{
    return py;
}

const std::vector<unsigned int> particles_4d::get_steps() const
{
    return steps;
}

const std::vector<uint8_t> particles_4d::get_valid() const
{
    return valid;
}

const std::vector<uint8_t> particles_4d::get_ghost() const
{
    return ghost;
}

const std::vector<size_t> particles_4d::get_idx() const
{
    return idx;
}

const std::vector<size_t> particles_4d::get_idx_base() const
{
    return idx_base;
}

const size_t &particles_4d::get_n_particles() const
{
    return n_particles;
}

const size_t &particles_4d::get_n_ghosts_per_particle() const
{
    return n_ghosts_per_particle;
}


void particles_4d_gpu::_general_cudaMalloc()
{
    // allocate cuda memory
    cudaMalloc(&d_x, x.size() * sizeof(double));
    cudaMalloc(&d_px, px.size() * sizeof(double));
    cudaMalloc(&d_y, y.size() * sizeof(double));
    cudaMalloc(&d_py, py.size() * sizeof(double));

    cudaMalloc(&d_steps, steps.size() * sizeof(unsigned int));
    cudaMalloc(&d_valid, valid.size() * sizeof(bool));
    cudaMalloc(&d_ghost, ghost.size() * sizeof(bool));

    cudaMalloc(&d_idx, idx.size() * sizeof(size_t));
    cudaMalloc(&d_idx_base, idx_base.size() * sizeof(size_t));

    cudaMalloc(&d_rng_state, 512 * _optimal_nblocks() * sizeof(curandState));
    setup_random_states<<<_optimal_nblocks(), 512>>>(d_rng_state, clock());
}

void particles_4d_gpu::_general_cudaFree()
{
    // free cuda memory
    cudaFree(d_x);
    cudaFree(d_px);
    cudaFree(d_y);
    cudaFree(d_py);

    cudaFree(d_steps);
    cudaFree(d_valid);
    cudaFree(d_ghost);

    cudaFree(d_idx);
    cudaFree(d_idx_base);

    cudaFree(d_rng_state);
}

void particles_4d_gpu::_general_host_to_device_copy()
{
    cudaMemcpy(d_x, x.data(), x.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_px, px.data(), px.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y.data(), y.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_py, py.data(), py.size() * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_steps, steps.data(), steps.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_valid, valid.data(), valid.size() * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ghost, ghost.data(), ghost.size() * sizeof(uint8_t), cudaMemcpyHostToDevice);

    cudaMemcpy(d_idx, idx.data(), idx.size() * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_idx_base, idx_base.data(), idx_base.size() * sizeof(size_t), cudaMemcpyHostToDevice);
}

void particles_4d_gpu::_general_device_to_host_copy()
{
    cudaMemcpy(x.data(), d_x, x.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(px.data(), d_px, px.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(y.data(), d_y, y.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(py.data(), d_py, py.size() * sizeof(double), cudaMemcpyDeviceToHost);

    cudaMemcpy(steps.data(), d_steps, steps.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(valid.data(), d_valid, valid.size() * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(ghost.data(), d_ghost, ghost.size() * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    cudaMemcpy(idx.data(), d_idx, idx.size() * sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(idx_base.data(), d_idx_base, idx_base.size() * sizeof(size_t), cudaMemcpyDeviceToHost);
}

size_t particles_4d_gpu::_optimal_nblocks() const
{
    size_t n_threads = 512;
    return (x.size() + n_threads - 1) / n_threads;
}

particles_4d_gpu::particles_4d_gpu(
    const std::vector<double> &x_,
    const std::vector<double> &px_,
    const std::vector<double> &y_,
    const std::vector<double> &py_)
    : particles_4d(x_, px_, y_, py_)
{
    _general_cudaMalloc();
    _general_host_to_device_copy();
    
}

void particles_4d_gpu::reset()
{
    this->particles_4d::reset();

    // copy data to device
    _general_host_to_device_copy();
}

void particles_4d_gpu::add_ghost(const double &displacement_module, const std::string &direction)
{
    this->particles_4d::add_ghost(displacement_module, direction);

    // copy data to device
    _general_cudaFree();
    _general_cudaMalloc();
    _general_host_to_device_copy();
}

__global__ void gpu_particle_renormalize(
    double *x, double *px, double *y, double *py, const size_t *idx, const size_t *idx_base, const uint8_t *ghost, const size_t size, const double module)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= size)
        return;
    
    if (ghost[i] == false)
        return;

    double init_module = displacement(
        x[i], px[i], y[i], py[i],
        x[idx_base[i]], px[idx_base[i]], y[idx_base[i]], py[idx_base[i]]);

    realign(
        x[i], px[i], y[i], py[i],
        x[idx_base[i]], px[idx_base[i]], y[idx_base[i]], py[idx_base[i]],
        init_module, module);
}

void particles_4d_gpu::renormalize(const double &module_target)
{
    gpu_particle_renormalize<<<_optimal_nblocks(), 512>>>(
        d_x, d_px, d_y, d_py, d_idx, d_idx_base, d_ghost, x.size(), module_target);
}

__global__ void gpu_particle_displacements(
    const double *x, const double *px, const double *y, const double *py, double *out, const size_t *idx, const size_t *idx_base, const uint8_t *ghost, const size_t size, const size_t n_ghosts)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (n_ghosts == 0)
        return;

    if (i >= size)
        return;

    if (ghost[i] == false)
        return;

    size_t pad = size / (n_ghosts + 1);

    if (
        check_nan(x[i], px[i], y[i], py[i]) ||
        check_nan(x[idx_base[i]], px[idx_base[i]], y[idx_base[i]], py[idx_base[i]])
    )
    {
        out[idx_base[i] + pad * idx[i]] = NAN;
        return;
    }

    out[idx_base[i] + pad * idx[i]] = displacement(
        x[i], px[i], y[i], py[i],
        x[idx_base[i]], px[idx_base[i]], y[idx_base[i]], py[idx_base[i]]    
    );
}

const std::vector<std::vector<double>> particles_4d_gpu::get_displacement_module() const
{
    std::vector<std::vector<double>> val_displacement(n_ghosts_per_particle, std::vector<double>(n_particles, std::numeric_limits<double>::quiet_NaN()));

    double *d_out;
    cudaMalloc(&d_out, n_particles * n_ghosts_per_particle * sizeof(double));
    
    gpu_particle_displacements<<<_optimal_nblocks(), 512>>>(
        d_x, d_px, d_y, d_py, d_out, d_idx, d_idx_base, d_ghost, x.size(), n_ghosts_per_particle
    );

    for (size_t i = 0; i < n_ghosts_per_particle; i++)
    {
        cudaMemcpy(val_displacement[i].data(), d_out + i * n_particles, n_particles * sizeof(double), cudaMemcpyDeviceToHost);
    }

    std::vector<std::vector<double>> val_displacement_transposed(n_particles, std::vector<double>(n_ghosts_per_particle, std::numeric_limits<double>::quiet_NaN()));

    for (size_t i = 0; i < n_particles; i++)
    {
        for (size_t j = 0; j < n_ghosts_per_particle; j++)
        {
            val_displacement_transposed[i][j] = val_displacement[j][i];
        }
    }

    cudaFree(d_out);
    return val_displacement_transposed;
}

__global__ void gpu_particle_directions(
    const double *x, const double *px, const double *y, const double *py, double *out_x, double *out_px, double *out_y, double *out_py, const size_t *idx, const size_t *idx_base, const uint8_t *ghost, const size_t size, const size_t n_ghosts)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (n_ghosts == 0)
        return;

    if (i >= size)
        return;

    if (ghost[i] == false)
        return;

    size_t pad = size / (n_ghosts + 1);
    if (
        check_nan(x[i], px[i], y[i], py[i]) ||
        check_nan(x[idx_base[i]], px[idx_base[i]], y[idx_base[i]], py[idx_base[i]])
    )
    {
        out_x[idx_base[i] + pad * idx[i]] = NAN;
        out_px[idx_base[i] + pad * idx[i]] = NAN;
        out_y[idx_base[i] + pad * idx[i]] = NAN;
        out_py[idx_base[i] + pad * idx[i]] = NAN;
        return;
    }

    double initial_module = displacement(
        x[i], px[i], y[i], py[i],
        x[idx_base[i]], px[idx_base[i]], y[idx_base[i]], py[idx_base[i]]);
    
    out_x[idx_base[i] + pad * idx[i]] = x[i];
    out_px[idx_base[i] + pad * idx[i]] = px[i];
    out_y[idx_base[i] + pad * idx[i]] = y[i];
    out_py[idx_base[i] + pad * idx[i]] = py[i];

    realign(
        out_x[idx_base[i] + pad * idx[i]], out_px[idx_base[i] + pad * idx[i]],
        out_y[idx_base[i] + pad * idx[i]], out_py[idx_base[i] + pad * idx[i]],
        x[idx_base[i]], px[idx_base[i]], y[idx_base[i]], py[idx_base[i]], initial_module, 1.0
    );

    out_x[idx_base[i] + pad * idx[i]] -= x[idx_base[i]];
    out_px[idx_base[i] + pad * idx[i]] -= px[idx_base[i]];
    out_y[idx_base[i] + pad * idx[i]] -= y[idx_base[i]];
    out_py[idx_base[i] + pad * idx[i]] -= py[idx_base[i]];
}

const std::vector<std::vector<std::vector<double>>> particles_4d_gpu::get_displacement_direction() const
{
    std::vector<std::vector<std::vector<double>>> dir_displacement(
        {{std::vector<std::vector<double>>(n_ghosts_per_particle, std::vector<double>(n_particles, std::numeric_limits<double>::quiet_NaN())),
          std::vector<std::vector<double>>(n_ghosts_per_particle, std::vector<double>(n_particles, std::numeric_limits<double>::quiet_NaN())),
          std::vector<std::vector<double>>(n_ghosts_per_particle, std::vector<double>(n_particles, std::numeric_limits<double>::quiet_NaN())),
          std::vector<std::vector<double>>(n_ghosts_per_particle, std::vector<double>(n_particles, std::numeric_limits<double>::quiet_NaN()))}});

    double *d_out_x, *d_out_px, *d_out_y, *d_out_py;
    cudaMalloc(&d_out_x, n_particles * n_ghosts_per_particle * sizeof(double));
    cudaMalloc(&d_out_px, n_particles * n_ghosts_per_particle * sizeof(double));
    cudaMalloc(&d_out_y, n_particles * n_ghosts_per_particle * sizeof(double));
    cudaMalloc(&d_out_py, n_particles * n_ghosts_per_particle * sizeof(double));

    gpu_particle_directions<<<_optimal_nblocks(), 512>>>(
        d_x, d_px, d_y, d_py, d_out_x, d_out_px, d_out_y, d_out_py, d_idx, d_idx_base, d_ghost, x.size(), n_ghosts_per_particle
    );

    for (size_t i = 0; i < n_ghosts_per_particle; i++)
    {
        cudaMemcpy(dir_displacement[0][i].data(), d_out_x + i * n_particles, n_particles * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(dir_displacement[1][i].data(), d_out_px + i * n_particles, n_particles * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(dir_displacement[2][i].data(), d_out_y + i * n_particles, n_particles * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(dir_displacement[3][i].data(), d_out_py + i * n_particles, n_particles * sizeof(double), cudaMemcpyDeviceToHost);
    }

    cudaFree(d_out_x);
    cudaFree(d_out_px);
    cudaFree(d_out_y);
    cudaFree(d_out_py);

    std::vector<std::vector<std::vector<double>>> dir_displacement_transposed(
        {{std::vector<std::vector<double>>(n_particles, std::vector<double>(n_ghosts_per_particle, std::numeric_limits<double>::quiet_NaN())),
          std::vector<std::vector<double>>(n_particles, std::vector<double>(n_ghosts_per_particle, std::numeric_limits<double>::quiet_NaN())),
          std::vector<std::vector<double>>(n_particles, std::vector<double>(n_ghosts_per_particle, std::numeric_limits<double>::quiet_NaN())),
          std::vector<std::vector<double>>(n_particles, std::vector<double>(n_ghosts_per_particle, std::numeric_limits<double>::quiet_NaN()))}});
    
    for (size_t i = 0; i < n_ghosts_per_particle; i++)
    {
        for (size_t j = 0; j < n_particles; j++)
        {
            dir_displacement_transposed[0][j][i] = dir_displacement[0][i][j];
            dir_displacement_transposed[1][j][i] = dir_displacement[1][i][j];
            dir_displacement_transposed[2][j][i] = dir_displacement[2][i][j];
            dir_displacement_transposed[3][j][i] = dir_displacement[3][i][j];
        }
    }

    return dir_displacement_transposed;
}

const std::vector<double> particles_4d_gpu::get_x() const
{
    std::vector<double> x_copy(x.size());
    // copy data to host
    cudaMemcpy(x_copy.data(), d_x, x.size() * sizeof(double), cudaMemcpyDeviceToHost);
    return x_copy;
}

const std::vector<double> particles_4d_gpu::get_px() const
{
    std::vector<double> px_copy(px.size());
    // copy data to host
    cudaMemcpy(px_copy.data(), d_px, px.size() * sizeof(double), cudaMemcpyDeviceToHost);
    return px_copy;
}

const std::vector<double> particles_4d_gpu::get_y() const
{
    std::vector<double> y_copy(y.size());
    // copy data to host
    cudaMemcpy(y_copy.data(), d_y, y.size() * sizeof(double), cudaMemcpyDeviceToHost);
    return y_copy;
}

const std::vector<double> particles_4d_gpu::get_py() const
{
    std::vector<double> py_copy(py.size());
    // copy data to host
    cudaMemcpy(py_copy.data(), d_py, py.size() * sizeof(double), cudaMemcpyDeviceToHost);
    return py_copy;
}

const std::vector<unsigned int> particles_4d_gpu::get_steps() const
{
    std::vector<unsigned int> steps_copy(steps.size());
    // copy data to host
    cudaMemcpy(steps_copy.data(), d_steps, steps.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    return steps_copy;
}

const std::vector<uint8_t> particles_4d_gpu::get_valid() const
{
    std::vector<uint8_t> valid_copy(valid.size());
    // copy data to host
    cudaMemcpy(valid_copy.data(), d_valid, valid.size() * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    return valid_copy;
}

const std::vector<uint8_t> particles_4d_gpu::get_ghost() const
{
    std::vector<uint8_t> ghost_copy(ghost.size());
    // copy data to host
    cudaMemcpy(ghost_copy.data(), d_ghost, ghost.size() * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    return ghost_copy;
}

const std::vector<size_t> particles_4d_gpu::get_idx() const
{
    std::vector<size_t> idx_copy(idx.size());
    // copy data to host
    cudaMemcpy(idx_copy.data(), d_idx, idx.size() * sizeof(size_t), cudaMemcpyDeviceToHost);
    return idx_copy;
}

const std::vector<size_t> particles_4d_gpu::get_idx_base() const
{
    std::vector<size_t> idx_base_copy(idx_base.size());
    // copy data to host
    cudaMemcpy(idx_base_copy.data(), d_idx_base, idx_base.size() * sizeof(size_t), cudaMemcpyDeviceToHost);
    return idx_base_copy;
}

const size_t &particles_4d_gpu::get_n_particles() const
{
    return n_particles;
}

const size_t &particles_4d_gpu::get_n_ghosts_per_particle() const
{
    return n_ghosts_per_particle;
}

particles_4d_gpu::~particles_4d_gpu()
{
    _general_cudaFree();
}


matrix_4d_vector::matrix_4d_vector(size_t N)
{
    matrix = std::vector<std::vector<std::vector<double>>>(N, std::vector<std::vector<double>>(4, std::vector<double>(4, 0.0)));

    // initialize with identity matrices
    for (size_t i = 0; i < N; i++)
    {
        matrix[i][0][0] = 1.0;
        matrix[i][1][1] = 1.0;
        matrix[i][2][2] = 1.0;
        matrix[i][3][3] = 1.0;
    }
}

void matrix_4d_vector::reset()
{
    // reset to identity matrices
    matrix = std::vector<std::vector<std::vector<double>>>(matrix.size(), std::vector<std::vector<double>>(4, std::vector<double>(4, 0.0)));
    for (size_t i = 0; i < matrix.size(); i++)
    {
        matrix[i][0][0] = 1.0;
        matrix[i][1][1] = 1.0;
        matrix[i][2][2] = 1.0;
        matrix[i][3][3] = 1.0;
    }
}

void matrix_4d_vector::multiply(const std::vector<std::vector<std::vector<double>>> &l_matrix)
{
    assert(matrix.size() == l_matrix.size());
    auto n_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;

    for (size_t i = 0; i < n_threads; i++)
    {
        threads.push_back(std::thread([&, i]() {
            for (size_t j = i; j < matrix.size(); j += n_threads)
            {
                matrix[j] = multiply_matrices(l_matrix[j], matrix[j]);
            }
        }));
    }

    for (auto &thread : threads)
    {
        thread.join();
    }
}

void matrix_4d_vector::structured_multiply(const henon_tracker &tracker, const particles_4d &particles, const double &mu, const bool &reverse)
{
    return multiply(
        tracker.get_tangent_matrix(particles, mu, reverse));
}

void matrix_4d_vector::structured_multiply(const henon_tracker_gpu &tracker, const particles_4d_gpu &particles, const double &mu, const bool &reverse)
{
    return multiply(
        tracker.get_tangent_matrix(particles, mu, reverse));
}

const std::vector<std::vector<std::vector<double>>> &matrix_4d_vector::get_matrix() const
{
    return matrix;
}

std::vector<std::vector<double>> matrix_4d_vector::get_vector(const std::vector<std::vector<double>> &rv) const
{
    std::vector<std::vector<double>> vectors(matrix.size(), std::vector<double>(4, 0.0));

    auto n_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;

    for (size_t i = 0; i < n_threads; i++)
    {
        threads.push_back(std::thread([&, i]()
                                      {
            for (size_t j = i; j < matrix.size(); j += n_threads)
            {
                // multiply matrix[j] with rv
                vectors[j][0] = matrix[j][0][0] * rv[j][0] + matrix[j][0][1] * rv[j][1] + matrix[j][0][2] * rv[j][2] + matrix[j][0][3] * rv[j][3];
                vectors[j][1] = matrix[j][1][0] * rv[j][0] + matrix[j][1][1] * rv[j][1] + matrix[j][1][2] * rv[j][2] + matrix[j][1][3] * rv[j][3];
                vectors[j][2] = matrix[j][2][0] * rv[j][0] + matrix[j][2][1] * rv[j][1] + matrix[j][2][2] * rv[j][2] + matrix[j][2][3] * rv[j][3];
                vectors[j][3] = matrix[j][3][0] * rv[j][0] + matrix[j][3][1] * rv[j][1] + matrix[j][3][2] * rv[j][2] + matrix[j][3][3] * rv[j][3];
            } }));
    }

    for (auto &thread : threads)
    {
        thread.join();
    }

    return vectors;
}

matrix_4d_vector_gpu::matrix_4d_vector_gpu(size_t _N)
{
    N = _N;
    n_blocks = (N + 512 - 1) / 512;
    std::vector<double> matrix_flattened(N * 16, 0.0);
    for (size_t i = 0; i < N; i++)
    {
        matrix_flattened[i * 16 + 0] = 1.0;
        matrix_flattened[i * 16 + 5] = 1.0;
        matrix_flattened[i * 16 + 10] = 1.0;
        matrix_flattened[i * 16 + 15] = 1.0;
    }

    // allocate memory on GPU
    cudaMalloc(&d_matrix, matrix_flattened.size() * sizeof(double));
    cudaMemcpy(d_matrix, matrix_flattened.data(), matrix_flattened.size() * sizeof(double), cudaMemcpyHostToDevice);
}

matrix_4d_vector_gpu::~matrix_4d_vector_gpu()
{
    cudaFree(d_matrix);
}

void matrix_4d_vector_gpu::reset()
{
    // reset to identity matrices
    std::vector<double> matrix_flattened(matrix_flattened.size(), 0.0);
    for (size_t i = 0; i < matrix_flattened.size(); i += 16)
    {
        matrix_flattened[i + 0] = 1.0;
        matrix_flattened[i + 5] = 1.0;
        matrix_flattened[i + 10] = 1.0;
        matrix_flattened[i + 15] = 1.0;
    }

    cudaMemcpy(d_matrix, matrix_flattened.data(), matrix_flattened.size() * sizeof(double), cudaMemcpyHostToDevice);
}

void matrix_4d_vector_gpu::structured_multiply(const henon_tracker_gpu &tracker, const particles_4d_gpu &particles, const double &mu)
{
    get_matrices_and_multiply<<<n_blocks, 512>>>(
        d_matrix,
        particles.d_x, particles.d_px, particles.d_y, particles.d_py, 
        tracker.d_omega_x_sin, tracker.d_omega_x_cos,
        tracker.d_omega_y_sin, tracker.d_omega_y_cos,
        mu, particles.global_steps, N);
}

void matrix_4d_vector_gpu::set_with_tracker(const henon_tracker_gpu &tracker, const particles_4d_gpu &particles, const double &mu)
{
    get_matrices_and_set<<<n_blocks, 512>>>(
        d_matrix,
        particles.d_x, particles.d_px, particles.d_y, particles.d_py, 
        tracker.d_omega_x_sin, tracker.d_omega_x_cos,
        tracker.d_omega_y_sin, tracker.d_omega_y_cos,
        mu, particles.global_steps, N);
}

std::vector<std::vector<double>> matrix_4d_vector_gpu::get_vector(const std::vector<std::vector<double>> &rv) const
{
    std::vector<double> matrix_flattened(N * 16, 0.0);

    // copy matrix from GPU to CPU
    cudaMemcpy(matrix_flattened.data(), d_matrix, matrix_flattened.size() * sizeof(double), cudaMemcpyDeviceToHost);

    std::vector<std::vector<std::vector<double>>> matrix(N, std::vector<std::vector<double>>(4, std::vector<double>(4, 0.0)));

    for (size_t i = 0; i < N; i++)
    {
        matrix[i][0][0] = matrix_flattened[i * 16 + 0];
        matrix[i][0][1] = matrix_flattened[i * 16 + 1];
        matrix[i][0][2] = matrix_flattened[i * 16 + 2];
        matrix[i][0][3] = matrix_flattened[i * 16 + 3];

        matrix[i][1][0] = matrix_flattened[i * 16 + 4];
        matrix[i][1][1] = matrix_flattened[i * 16 + 5];
        matrix[i][1][2] = matrix_flattened[i * 16 + 6];
        matrix[i][1][3] = matrix_flattened[i * 16 + 7];

        matrix[i][2][0] = matrix_flattened[i * 16 + 8];
        matrix[i][2][1] = matrix_flattened[i * 16 + 9];
        matrix[i][2][2] = matrix_flattened[i * 16 + 10];
        matrix[i][2][3] = matrix_flattened[i * 16 + 11];

        matrix[i][3][0] = matrix_flattened[i * 16 + 12];
        matrix[i][3][1] = matrix_flattened[i * 16 + 13];
        matrix[i][3][2] = matrix_flattened[i * 16 + 14];
        matrix[i][3][3] = matrix_flattened[i * 16 + 15];
    }

    std::vector<std::vector<double>> vectors(matrix.size(), std::vector<double>(4, 0.0));

    auto n_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;

    for (size_t i = 0; i < n_threads; i++)
    {
        threads.push_back(std::thread([&, i]()
                                      {
            for (size_t j = i; j < matrix.size(); j += n_threads)
            {
                // multiply matrix[j] with rv
                vectors[j][0] = matrix[j][0][0] * rv[j][0] + matrix[j][0][1] * rv[j][1] + matrix[j][0][2] * rv[j][2] + matrix[j][0][3] * rv[j][3];
                vectors[j][1] = matrix[j][1][0] * rv[j][0] + matrix[j][1][1] * rv[j][1] + matrix[j][1][2] * rv[j][2] + matrix[j][1][3] * rv[j][3];
                vectors[j][2] = matrix[j][2][0] * rv[j][0] + matrix[j][2][1] * rv[j][1] + matrix[j][2][2] * rv[j][2] + matrix[j][2][3] * rv[j][3];
                vectors[j][3] = matrix[j][3][0] * rv[j][0] + matrix[j][3][1] * rv[j][1] + matrix[j][3][2] * rv[j][2] + matrix[j][3][3] * rv[j][3];
            } }));
    }

    for (auto &thread : threads)
    {
        thread.join();
    }

    return vectors;
}

const std::vector<std::vector<std::vector<double>>> matrix_4d_vector_gpu::get_matrix() const
{
    std::vector<double> matrix_flattened(N * 16, 0.0);

    // copy matrix from GPU to CPU
    cudaMemcpy(matrix_flattened.data(), d_matrix, matrix_flattened.size() * sizeof(double), cudaMemcpyDeviceToHost);

    std::vector<std::vector<std::vector<double>>> matrix(N, std::vector<std::vector<double>>(4, std::vector<double>(4, 0.0)));

    for (size_t i = 0; i < N; i++)
    {
        matrix[i][0][0] = matrix_flattened[i * 16 + 0];
        matrix[i][0][1] = matrix_flattened[i * 16 + 1];
        matrix[i][0][2] = matrix_flattened[i * 16 + 2];
        matrix[i][0][3] = matrix_flattened[i * 16 + 3];

        matrix[i][1][0] = matrix_flattened[i * 16 + 4];
        matrix[i][1][1] = matrix_flattened[i * 16 + 5];
        matrix[i][1][2] = matrix_flattened[i * 16 + 6];
        matrix[i][1][3] = matrix_flattened[i * 16 + 7];

        matrix[i][2][0] = matrix_flattened[i * 16 + 8];
        matrix[i][2][1] = matrix_flattened[i * 16 + 9];
        matrix[i][2][2] = matrix_flattened[i * 16 + 10];
        matrix[i][2][3] = matrix_flattened[i * 16 + 11];

        matrix[i][3][0] = matrix_flattened[i * 16 + 12];
        matrix[i][3][1] = matrix_flattened[i * 16 + 13];
        matrix[i][3][2] = matrix_flattened[i * 16 + 14];
        matrix[i][3][3] = matrix_flattened[i * 16 + 15];
    }

    return matrix;
}

vector_4d_gpu::vector_4d_gpu(const std::vector<std::vector<double>> &vectors)
{
    N = vectors.size();
    n_blocks = (N + 512 - 1) / 512;

    // flatten the vector
    std::vector<double> vectors_flattened(N * 4, 0.0);

    for (size_t i = 0; i < N; i++)
    {
        vectors_flattened[i * 4 + 0] = vectors[i][0];
        vectors_flattened[i * 4 + 1] = vectors[i][1];
        vectors_flattened[i * 4 + 2] = vectors[i][2];
        vectors_flattened[i * 4 + 3] = vectors[i][3];
    }

    // allocate memory on GPU
    cudaMalloc(&d_vectors, N * 4 * sizeof(double));

    // copy vectors from CPU to GPU
    cudaMemcpy(d_vectors, vectors_flattened.data(), N * 4 * sizeof(double), cudaMemcpyHostToDevice);
}

vector_4d_gpu::~vector_4d_gpu()
{
    cudaFree(d_vectors);
}

void vector_4d_gpu::set_vectors(const std::vector<std::vector<double>> &vectors)
{
    // flatten the vector
    std::vector<double> vectors_flattened(N * 4, 0.0);

    for (size_t i = 0; i < N; i++)
    {
        vectors_flattened[i * 4 + 0] = vectors[i][0];
        vectors_flattened[i * 4 + 1] = vectors[i][1];
        vectors_flattened[i * 4 + 2] = vectors[i][2];
        vectors_flattened[i * 4 + 3] = vectors[i][3];
    }
    // copy vectors from CPU to GPU
    cudaMemcpy(d_vectors, vectors_flattened.data(), N * 4 * sizeof(double), cudaMemcpyHostToDevice);
}

void vector_4d_gpu::set_vectors(const std::vector<double> &vectors)
{
    // clone the vector over the flattened vector
    std::vector<double> vectors_flattened(N * 4, 0.0);

    for (size_t i = 0; i < N; i++)
    {
        vectors_flattened[i * 4 + 0] = vectors[0];
        vectors_flattened[i * 4 + 1] = vectors[1];
        vectors_flattened[i * 4 + 2] = vectors[2];
        vectors_flattened[i * 4 + 3] = vectors[3];
    }

    // copy vectors from CPU to GPU
    cudaMemcpy(d_vectors, vectors_flattened.data(), N * 4 * sizeof(double), cudaMemcpyHostToDevice);
}

void vector_4d_gpu::multiply(const matrix_4d_vector_gpu &matrix)
{
    // multiply matrix with vectors
    matrix_vector_multiply<<<n_blocks, 512>>>(matrix.d_matrix, d_vectors, N);
}

void vector_4d_gpu::normalize()
{
    // normalize vectors
    normalize_vectors<<<n_blocks, 512>>>(d_vectors, N);
}

const std::vector<std::vector<double>> vector_4d_gpu::get_vectors() const
{
    std::vector<double> vectors_flattened(N * 4, 0.0);

    // copy vectors from GPU to CPU
    cudaMemcpy(vectors_flattened.data(), d_vectors, N * 4 * sizeof(double), cudaMemcpyDeviceToHost);

    std::vector<std::vector<double>> vectors(N, std::vector<double>(4, 0.0));

    for (size_t i = 0; i < N; i++)
    {
        vectors[i][0] = vectors_flattened[i * 4 + 0];
        vectors[i][1] = vectors_flattened[i * 4 + 1];
        vectors[i][2] = vectors_flattened[i * 4 + 2];
        vectors[i][3] = vectors_flattened[i * 4 + 3];
    }

    return vectors;
}


lyapunov_birkhoff_construct::lyapunov_birkhoff_construct(size_t _N, size_t _n_weights): N(_N), n_weights(_n_weights), idx(0)
{
    n_blocks = (N + 512 - 1) / 512;

    auto birkhoff = birkhoff_weights(n_weights);
    // copy birkhoff to GPU
    cudaMalloc(&d_birkhoff, n_weights * sizeof(double));
    cudaMemcpy(d_birkhoff, birkhoff.data(), n_weights * sizeof(double), cudaMemcpyHostToDevice);

    // create a zero filled vector on GPU of size N
    cudaMalloc(&d_vector, N * sizeof(double));
    cudaMemset(d_vector, 0.0, N * sizeof(double));

    cudaMalloc(&d_vector_b, N * sizeof(double));
    cudaMemset(d_vector_b, 0.0, N * sizeof(double));
}

lyapunov_birkhoff_construct::~lyapunov_birkhoff_construct()
{
    cudaFree(d_birkhoff);
    cudaFree(d_vector);
    cudaFree(d_vector_b);
}

void lyapunov_birkhoff_construct::reset()
{
    idx = 0;
    cudaMemset(d_vector, 0.0, N * sizeof(double));
    cudaMemset(d_vector_b, 0.0, N * sizeof(double));
}
void lyapunov_birkhoff_construct::change_weights(size_t _n_weights)
{
    n_weights = _n_weights;
    auto birkhoff = birkhoff_weights(n_weights);
    // remove old weights
    cudaFree(d_birkhoff);
    // copy birkhoff to GPU
    cudaMalloc(&d_birkhoff, n_weights * sizeof(double));
    cudaMemcpy(d_birkhoff, birkhoff.data(), n_weights * sizeof(double), cudaMemcpyHostToDevice);
}

__global__ void kernel_lyap_birk(double *values, double *values_b, const double *vectors, const double *weights, size_t size, size_t n_weights, size_t b_idx)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;

    double norm = 0.0;
    for (size_t i = 0; i < 4; i++)
    {
        norm += vectors[idx*4 + i] * vectors[idx*4 + i];
    }
    norm = sqrt(norm);
    
    values[idx] += log(norm) / n_weights;
    values_b[idx] += log(norm) * weights[b_idx];
}

void lyapunov_birkhoff_construct::add(const vector_4d_gpu &vectors)
{
    // add vectors to lyapunov
    kernel_lyap_birk<<<n_blocks, 512>>>(d_vector, d_vector_b, vectors.d_vectors, d_birkhoff, N, n_weights, idx);
    idx++;
}

std::vector<double> lyapunov_birkhoff_construct::get_weights() const
{
    std::vector<double> weights(n_weights, 0.0);
    cudaMemcpy(weights.data(), d_birkhoff, n_weights * sizeof(double), cudaMemcpyDeviceToHost);
    return weights;
}

std::vector<double> lyapunov_birkhoff_construct::get_values_raw() const
{
    std::vector<double> values(N, 0.0);
    // copy values from GPU to CPU
    cudaMemcpy(values.data(), d_vector, N * sizeof(double), cudaMemcpyDeviceToHost);
    return values;
}

std::vector<double> lyapunov_birkhoff_construct::get_values_b() const
{
    std::vector<double> values(N, 0.0);
    // copy values from GPU to CPU
    cudaMemcpy(values.data(), d_vector_b, N * sizeof(double), cudaMemcpyDeviceToHost);
    return values;
}


void henon_tracker::compute_a_modulation(unsigned int n_turns, double omega_x, double omega_y, std::string modulation_kind, double omega_0, double epsilon, unsigned int offset)
{ 
    // get cpu threads
    unsigned int n_threads_cpu = std::thread::hardware_concurrency();
    // compute a modulation
    tie(omega_x_vec, omega_y_vec) = pick_a_modulation(n_turns, omega_x, omega_y, modulation_kind, omega_0, epsilon, offset);

    // copy to vectors
    omega_x_sin.resize(omega_x_vec.size());
    omega_x_cos.resize(omega_x_vec.size());
    omega_y_sin.resize(omega_y_vec.size());
    omega_y_cos.resize(omega_y_vec.size());

    std::vector<std::thread> threads;

    for (unsigned int k = 0; k < n_threads_cpu; k++)
    {
        threads.push_back(std::thread(
            [&](const unsigned int n_th)
            {
                for (size_t i = n_th; i < omega_x_vec.size(); i += n_threads_cpu)
                {
                    omega_x_sin[i] = sin(omega_x_vec[i]);
                    omega_x_cos[i] = cos(omega_x_vec[i]);
                    omega_y_sin[i] = sin(omega_y_vec[i]);
                    omega_y_cos[i] = cos(omega_y_vec[i]);
                }
            },
            k));
    }

    // join threads
    for (auto &t : threads)
    {
        t.join();
    }
    allowed_steps = n_turns;
}

henon_tracker::henon_tracker(unsigned int n_turns, double omega_x, double omega_y, std::string modulation_kind, double omega_0, double epsilon, unsigned int offset)
{
    compute_a_modulation(n_turns, omega_x, omega_y, modulation_kind, omega_0, epsilon, offset);   
}

void henon_tracker::track(particles_4d &particles, unsigned int n_turns, double mu, double barrier, double kick_module, bool inverse)
{
    unsigned int n_threads_cpu = std::thread::hardware_concurrency();

    // check if n_turns is valid
    if (inverse)
    {
        if (n_turns > particles.global_steps)
            throw std::runtime_error("The number of turns is too large.");
    }
    else
    {
        if (n_turns + particles.global_steps > allowed_steps)
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
                for (unsigned int j = thread_idx; j < particles.x.size(); j += n_threads_cpu)
                {
                    for (unsigned int k = 0; k < n_turns; k++)
                    {
                        henon_step(
                            particles.x[j], particles.px[j], particles.y[j], particles.py[j], particles.steps[j],
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
        particles.global_steps += n_turns;
    else
        particles.global_steps -= n_turns;
}

std::vector<std::vector<double>> henon_tracker::birkhoff_tunes(particles_4d &particles, unsigned int n_turns, double mu, double barrier, double kick_module, bool inverse, std::vector<unsigned int> from_idx, std::vector<unsigned int> to_idx)
{
    from_idx.push_back(0);
    to_idx.push_back(n_turns);
    std::vector<std::vector<double>> result_vec(particles.x.size());

    unsigned int n_threads_cpu = std::thread::hardware_concurrency();

    // check if n_turns is valid
    if (inverse)
    {
        if (n_turns > particles.global_steps)
            throw std::runtime_error("The number of turns is too large.");
    }
    else
    {
        if (n_turns + particles.global_steps > allowed_steps)
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
                std::vector<double> store_x(n_turns + 1);
                std::vector<double> store_px(n_turns + 1);
                std::vector<double> store_y(n_turns + 1);
                std::vector<double> store_py(n_turns + 1);

                std::mt19937_64 rng;

                for (unsigned int j = thread_idx; j < particles.x.size(); j += n_threads_cpu)
                {
                    store_x[0] = particles.x[j];
                    store_px[0] = particles.px[j];
                    store_y[0] = particles.y[j];
                    store_py[0] = particles.py[j];
                    for (unsigned int k = 0; k < n_turns; k++)
                    {
                        henon_step(
                            particles.x[j], particles.px[j], particles.y[j], particles.py[j], particles.steps[j],
                            omega_x_sin.data(), omega_x_cos.data(),
                            omega_y_sin.data(), omega_y_cos.data(),
                            barrier_pow_2, mu, kick_module,
                            rng, inverse);

                        store_x[k + 1] = particles.x[j];
                        store_px[k + 1] = particles.px[j];
                        store_y[k + 1] = particles.y[j];
                        store_py[k + 1] = particles.py[j];
                    }
                    auto result = birkhoff_tune_vec(store_x, store_px, store_y, store_py, from_idx, to_idx);
                    result_vec[j] = result;
                }
            },
            i));
    }

    // join threads
    for (auto &t : threads)
        t.join();

    // update global_steps
    if (!inverse)
        particles.global_steps += n_turns;
    else
        particles.global_steps -= n_turns;

    return result_vec;
}

std::vector<std::vector<double>> henon_tracker::all_tunes(particles_4d &particles, unsigned int n_turns, double mu, double barrier, double kick_module, bool inverse, std::vector<unsigned int> from_idx, std::vector<unsigned int> to_idx)
{
    from_idx.push_back(0);
    to_idx.push_back(n_turns);

    std::set<unsigned int> differences;
    for (unsigned int i = 0; i < from_idx.size(); i++)
    {
        differences.insert(to_idx[i] - from_idx[i] + 1);
        differences.insert(to_idx[i] - from_idx[i]);
    }
    // convert set to vector
    std::vector<unsigned int> differences_vec(differences.begin(), differences.end());

    std::vector<std::vector<double>> result_vec(particles.x.size());

    unsigned int n_threads_cpu = std::thread::hardware_concurrency();
    // unsigned int n_threads_cpu = 1;

    // check if n_turns is valid
    if (inverse)
    {
        if (n_turns > particles.global_steps)
            throw std::runtime_error("The number of turns is too large.");
    }
    else
    {
        if (n_turns + particles.global_steps > allowed_steps)
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
                std::vector<double> store_x(n_turns + 1);
                std::vector<double> store_px(n_turns + 1);
                std::vector<double> store_y(n_turns + 1);
                std::vector<double> store_py(n_turns + 1);

                std::mt19937_64 rng;
                auto fft_memory = fft_allocate(differences_vec);

                for (unsigned int j = thread_idx; j < particles.x.size(); j += n_threads_cpu)
                {
                    store_x[0] = particles.x[j];
                    store_px[0] = particles.px[j];
                    store_y[0] = particles.y[j];
                    store_py[0] = particles.py[j];
                    for (unsigned int k = 0; k < n_turns; k++)
                    {
                        henon_step(
                            particles.x[j], particles.px[j], particles.y[j], particles.py[j], particles.steps[j],
                            omega_x_sin.data(), omega_x_cos.data(),
                            omega_y_sin.data(), omega_y_cos.data(),
                            barrier_pow_2, mu, kick_module,
                            rng, inverse);

                        store_x[k + 1] = particles.x[j];
                        store_px[k + 1] = particles.px[j];
                        store_y[k + 1] = particles.y[j];
                        store_py[k + 1] = particles.py[j];
                    }
                    // remove the mean to store_x
                    double mean_x = std::accumulate(store_x.begin(), store_x.end(), 0.0) / store_x.size();
                    std::transform(store_x.begin(), store_x.end(), store_x.begin(), [mean_x](double x) { return x - mean_x; });
                    // remove the mean to store_px
                    double mean_px = std::accumulate(store_px.begin(), store_px.end(), 0.0) / store_px.size();
                    std::transform(store_px.begin(), store_px.end(), store_px.begin(), [mean_px](double x) { return x - mean_px; });
                    // remove the mean to store_y
                    double mean_y = std::accumulate(store_y.begin(), store_y.end(), 0.0) / store_y.size();
                    std::transform(store_y.begin(), store_y.end(), store_y.begin(), [mean_y](double x) { return x - mean_y; });
                    // remove the mean to store_py
                    double mean_py = std::accumulate(store_py.begin(), store_py.end(), 0.0) / store_py.size();

                    auto result_birk = birkhoff_tune_vec(store_x, store_px, store_y, store_py, from_idx, to_idx);
                    auto result_fft = fft_tune_vec(store_x, store_px, store_y, store_py, from_idx, to_idx, fft_memory);

                    // concatenate the two vectors
                    result_vec[j] = std::vector<double>(result_birk.size() + result_fft.size());
                    std::copy(result_birk.begin(), result_birk.end(), result_vec[j].begin());
                    std::copy(result_fft.begin(), result_fft.end(), result_vec[j].begin() + result_birk.size());
                }
                fft_free(fft_memory);
            },
            i));
    }

    // join threads
    for (auto &t : threads)
        t.join();
    // update global_steps
    if (!inverse)
        particles.global_steps += n_turns;
    else
        particles.global_steps -= n_turns;

    return result_vec;
}

std::vector<std::vector<std::vector<double>>> henon_tracker::get_tangent_matrix(const particles_4d &particles, const double &mu, const bool &reverse) const
{
    auto x = particles.get_x();
    auto px = particles.get_px();
    auto y = particles.get_y();
    auto py = particles.get_py();

    auto valid = particles.get_valid();
    auto steps = particles.global_steps;

    std::vector<std::vector<std::vector<double>>> result(x.size());

    for (unsigned int i = 0; i < x.size(); i++)
    {
        if (valid[i])
        {
            if (!reverse)
            {
                result[i] = tangent_matrix(x[i], px[i], y[i], py[i], omega_x_sin[steps], omega_x_cos[steps], omega_y_sin[steps], omega_y_cos[steps], mu);
            }
            else
            {
                result[i] = inverse_tangent_matrix(x[i], px[i], y[i], py[i], omega_x_sin[steps], omega_x_cos[steps], omega_y_sin[steps], omega_y_cos[steps], mu);
            }
        }
        else
        {
            result[i] = std::vector<std::vector<double>>(4, std::vector<double>(4, std::numeric_limits<double>::quiet_NaN()));
        }
    }

    return result;
}

henon_tracker_gpu::henon_tracker_gpu(unsigned int n_turns, double omega_x, double omega_y, std::string modulation_kind, double omega_0, double epsilon, unsigned int offset) : henon_tracker(n_turns, omega_x, omega_y, modulation_kind, omega_0, epsilon, offset) 
{
    cudaMalloc(&d_omega_x_sin, omega_x_sin.size() * sizeof(double));
    cudaMalloc(&d_omega_x_cos, omega_x_cos.size() * sizeof(double));
    cudaMalloc(&d_omega_y_sin, omega_y_sin.size() * sizeof(double));
    cudaMalloc(&d_omega_y_cos, omega_y_cos.size() * sizeof(double));

    cudaMemcpy(d_omega_x_sin, omega_x_sin.data(), omega_x_sin.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_omega_x_cos, omega_x_cos.data(), omega_x_cos.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_omega_y_sin, omega_y_sin.data(), omega_y_sin.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_omega_y_cos, omega_y_cos.data(), omega_y_cos.size() * sizeof(double), cudaMemcpyHostToDevice);
}

henon_tracker_gpu::~henon_tracker_gpu()
{
    cudaFree(d_omega_x_sin);
    cudaFree(d_omega_x_cos);
    cudaFree(d_omega_y_sin);
    cudaFree(d_omega_y_cos);
}

void henon_tracker_gpu::compute_a_modulation(unsigned int n_turns, double omega_x, double omega_y, std::string modulation_kind, double omega_0, double epsilon, unsigned int offset)
{
    henon_tracker::compute_a_modulation(n_turns, omega_x, omega_y, modulation_kind, omega_0, epsilon, offset);

    cudaFree(d_omega_x_sin);
    cudaFree(d_omega_x_cos);
    cudaFree(d_omega_y_sin);
    cudaFree(d_omega_y_cos);

    cudaMalloc(&d_omega_x_sin, omega_x_sin.size() * sizeof(double));
    cudaMalloc(&d_omega_x_cos, omega_x_cos.size() * sizeof(double));
    cudaMalloc(&d_omega_y_sin, omega_y_sin.size() * sizeof(double));
    cudaMalloc(&d_omega_y_cos, omega_y_cos.size() * sizeof(double));

    cudaMemcpy(d_omega_x_sin, omega_x_sin.data(), omega_x_sin.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_omega_x_cos, omega_x_cos.data(), omega_x_cos.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_omega_y_sin, omega_y_sin.data(), omega_y_sin.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_omega_y_cos, omega_y_cos.data(), omega_y_cos.size() * sizeof(double), cudaMemcpyHostToDevice);
}

void henon_tracker_gpu::track(particles_4d_gpu &particles, unsigned int n_turns, double mu, double barrier, double kick_module, bool inverse)
{
    // check if n_turns is valid
    if (inverse)
    {
        if (n_turns > particles.global_steps)
            throw std::runtime_error("The number of turns is too large.");
    }
    else
    {
        if (n_turns + particles.global_steps > allowed_steps)
            throw std::runtime_error("The number of turns is too large.");
    }

    for (unsigned int j = 0; j < n_turns; j++)
    {
        gpu_henon_track<<<particles._optimal_nblocks(), 512>>>(
            particles.d_x, particles.d_px, particles.d_y, particles.d_py, particles.d_steps,
            particles.x.size(), 1, barrier * barrier, mu,
            d_omega_x_sin, d_omega_x_cos, d_omega_y_sin, d_omega_y_cos,
            kick_module, particles.d_rng_state, inverse);

        if (j % 1000 == 0)
            check_cuda_errors();
    }
    // Update the counter
    if (!inverse)
        particles.global_steps += n_turns;
    else
        particles.global_steps -= n_turns;
}

std::vector<std::vector<std::vector<double>>> henon_tracker_gpu::get_tangent_matrix(const particles_4d_gpu &particles, const double &mu, const bool &reverse) const
{
    auto x = particles.get_x();
    auto px = particles.get_px();
    auto y = particles.get_y();
    auto py = particles.get_py();

    auto valid = particles.get_valid();
    auto steps = particles.global_steps;
    
    std::vector<std::vector<std::vector<double>>> result(x.size());

    for (unsigned int i = 0; i < x.size(); i++)
    {
        if (valid[i])
        {
            if (!reverse)
            {
                result[i] = tangent_matrix(x[i], px[i], y[i], py[i], omega_x_sin[steps], omega_x_cos[steps], omega_y_sin[steps], omega_y_cos[steps], mu);
            }
            else
            {
                result[i] = inverse_tangent_matrix(x[i], px[i], y[i], py[i], omega_x_sin[steps], omega_x_cos[steps], omega_y_sin[steps], omega_y_cos[steps], mu);
            }
        }
        else
        {
            result[i] = std::vector<std::vector<double>>(4, std::vector<double>(4, std::numeric_limits<double>::quiet_NaN()));
        }
    }

    return result;
}

storage_4d::storage_4d(size_t N)
{
    x.resize(N);
    px.resize(N);
    y.resize(N);
    py.resize(N);
}

void storage_4d::store(const particles_4d &particles)
{
    auto tmp_x = particles.get_x();
    auto tmp_px = particles.get_px();
    auto tmp_y = particles.get_y();
    auto tmp_py = particles.get_py();

    for (size_t i = 0; i < x.size(); i++)
    {
        x[i].push_back(tmp_x[i]);
        px[i].push_back(tmp_px[i]);
        y[i].push_back(tmp_y[i]);
        py[i].push_back(tmp_py[i]);
    }
}

void storage_4d::store(const particles_4d_gpu &particles)
{
    auto tmp_x = particles.get_x();
    auto tmp_px = particles.get_px();
    auto tmp_y = particles.get_y();
    auto tmp_py = particles.get_py();

    for (size_t i = 0; i < x.size(); i++)
    {
        x[i].push_back(tmp_x[i]);
        px[i].push_back(tmp_px[i]);
        y[i].push_back(tmp_y[i]);
        py[i].push_back(tmp_py[i]);
    }
}

std::vector<std::vector<double>> storage_4d::tune_fft(std::vector<unsigned int> from, std::vector<unsigned int> to) const
{
    unsigned int n_threads_cpu = std::thread::hardware_concurrency();

    auto n_turns = x[0].size();
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

    std::vector<std::thread> threads;

    std::vector<std::vector<double>> result_vec(x.size());
    for (unsigned int i = 0; i < n_threads_cpu; i++)
    {
        threads.push_back(std::thread(
            [&](int thread_idx)
            {
                // create plans
                auto plans = fft_allocate(n_list);

                for (unsigned int j = thread_idx; j < x.size(); j += n_threads_cpu)
                {
                    auto result = fft_tune_vec(x[j], px[j], y[j], py[j], from, to, plans);
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

    return result_vec;
}

std::vector<std::vector<double>> storage_4d::tune_birkhoff(std::vector<unsigned int> from, std::vector<unsigned int> to) const
{
    unsigned int n_threads_cpu = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    std::vector<std::vector<double>> result_vec(x.size());
    for (unsigned int i = 0; i < n_threads_cpu; i++)
    {
        threads.push_back(std::thread(
            [&](int thread_idx)
            {
                for (unsigned int j = thread_idx; j < x.size(); j += n_threads_cpu)
                {
                    auto result = birkhoff_tune_vec(x[j], px[j], y[j], py[j], from, to);
                    result_vec[j] = result;
                }
            },
            i));
    }
    // join threads
    for (auto &t : threads)
        t.join();
    return result_vec;
}

const std::vector<std::vector<double>> &storage_4d::get_x() const
{
    return x;
}

const std::vector<std::vector<double>> &storage_4d::get_px() const
{
    return px;
}

const std::vector<std::vector<double>> &storage_4d::get_y() const
{
    return y;
}

const std::vector<std::vector<double>> &storage_4d::get_py() const
{
    return py;
}

storage_4d_gpu::storage_4d_gpu(size_t _N, size_t _batch_size)
{
    N = _N;
    batch_size = _batch_size;
    n_blocks = (N + 1023) / 1024;
    idx = 0;
    // allocate memory
    cudaMalloc(&d_x, N * batch_size * sizeof(double));
    cudaMalloc(&d_px, N * batch_size * sizeof(double));
    cudaMalloc(&d_y, N * batch_size * sizeof(double));
    cudaMalloc(&d_py, N * batch_size * sizeof(double));
    // set all to NaN
    cudaMemset(d_x, 0, N * batch_size * sizeof(double));
    cudaMemset(d_px, 0, N * batch_size * sizeof(double));
    cudaMemset(d_y, 0, N * batch_size * sizeof(double));
    cudaMemset(d_py, 0, N * batch_size * sizeof(double));
}

storage_4d_gpu::~storage_4d_gpu()
{
    cudaFree(d_x);
    cudaFree(d_px);
    cudaFree(d_y);
    cudaFree(d_py);
}

void storage_4d_gpu::store(const particles_4d_gpu &particles)
{
    cudaMemcpy(d_x + idx * N, particles.d_x, N * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_px + idx * N, particles.d_px, N * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_y + idx * N, particles.d_y, N * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_py + idx * N, particles.d_py, N * sizeof(double), cudaMemcpyDeviceToDevice);

    idx++;
}

void storage_4d_gpu::reset()
{
    idx = 0;
    // set all to NaN
    cudaMemset(d_x, 0, N * batch_size * sizeof(double));
    cudaMemset(d_px, 0, N * batch_size * sizeof(double));
    cudaMemset(d_y, 0, N * batch_size * sizeof(double));
    cudaMemset(d_py, 0, N * batch_size * sizeof(double));
}

const std::vector<std::vector<double>> storage_4d_gpu::get_x() const
{
    std::vector<std::vector<double>> result(batch_size, std::vector<double>(N));
    std::vector<double> temp(N * batch_size, 0);
    cudaMemcpy(temp.data(), d_x, N * batch_size * sizeof(double), cudaMemcpyDeviceToHost);
    
    for (unsigned int i = 0; i < batch_size; i++)
    {
        for (unsigned int j = 0; j < N; j++)
        {
            result[i][j] = temp[i * N + j];
        }
    }

    return result;
}

const std::vector<std::vector<double>> storage_4d_gpu::get_px() const
{
    std::vector<std::vector<double>> result(batch_size, std::vector<double>(N));
    std::vector<double> temp(N * batch_size, 0);
    cudaMemcpy(temp.data(), d_px, N * batch_size * sizeof(double), cudaMemcpyDeviceToHost);
    
    for (unsigned int i = 0; i < batch_size; i++)
    {
        for (unsigned int j = 0; j < N; j++)
        {
            result[i][j] = temp[i * N + j];
        }
    }

    return result;
}

const std::vector<std::vector<double>> storage_4d_gpu::get_y() const
{
    std::vector<std::vector<double>> result(batch_size, std::vector<double>(N));
    std::vector<double> temp(N * batch_size, 0);
    cudaMemcpy(temp.data(), d_y, N * batch_size * sizeof(double), cudaMemcpyDeviceToHost);
    
    for (unsigned int i = 0; i < batch_size; i++)
    {
        for (unsigned int j = 0; j < N; j++)
        {
            result[i][j] = temp[i * N + j];
        }
    }

    return result;
}

const std::vector<std::vector<double>> storage_4d_gpu::get_py() const
{
    std::vector<std::vector<double>> result(batch_size, std::vector<double>(N));
    std::vector<double> temp(N * batch_size, 0);
    cudaMemcpy(temp.data(), d_py, N * batch_size * sizeof(double), cudaMemcpyDeviceToHost);
    
    for (unsigned int i = 0; i < batch_size; i++)
    {
        for (unsigned int j = 0; j < N; j++)
        {
            result[i][j] = temp[i * N + j];
        }
    }

    return result;
}