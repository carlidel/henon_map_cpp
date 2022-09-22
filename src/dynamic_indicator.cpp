#include "dynamic_indicator.h"

// mutex lock for fft
std::mutex fft_lock;

std::map<unsigned int, std::tuple<fftw_complex *, fftw_complex *, fftw_plan>> fft_allocate(std::vector<unsigned int> const &n_points)
{
    fft_lock.lock();
    std::map<unsigned int, std::tuple<fftw_complex *, fftw_complex *, fftw_plan>> plans;
    for (auto &x: n_points) {
        fftw_complex *in = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * x);
        fftw_complex *out = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * x);
        fftw_plan plan = fftw_plan_dft_1d(x, in, out, FFTW_FORWARD, FFTW_MEASURE);
        plans[x] = std::make_tuple(in, out, plan);
    }
    fft_lock.unlock();
    return plans;
}

void fft_free(std::map<unsigned int, std::tuple<fftw_complex *, fftw_complex *, fftw_plan>> &fft_map)
{
    fft_lock.lock();
    for (auto &x: fft_map) {
        fftw_destroy_plan(std::get<2>(x.second));
        fftw_free(std::get<0>(x.second));
        fftw_free(std::get<1>(x.second));
    }
    fft_map.clear();
    fft_lock.unlock();
}

std::vector<double> compute_fft(std::vector<double> const &real_signal, std::vector<double> const &imag_signal, bool hanning_window, fftw_complex *in, fftw_complex *out, fftw_plan p)
{
    // check if the signal is of the correct length
    if (real_signal.size() != imag_signal.size()) {
        throw std::invalid_argument("real and imag signals must have the same length");
    }
    auto N = real_signal.size();
    
    std::vector<double> fft_module(N);

    for (unsigned int i = 0; i < N; i++) {
        in[i][0] = real_signal[i];
        in[i][1] = imag_signal[i];
    }

    // if hanning window is true, apply hanning window
    if (hanning_window) {
        for (unsigned int i = 0; i < N; i++) {
            in[i][0] *= 2 * pow((sin(M_PI * i / (N))), 2);
            in[i][1] *= 2 * pow((sin(M_PI * i / (N))), 2);
        }
    }

    fftw_execute(p);
    for (unsigned int i = 0; i < N; i++) {
        fft_module[i] = sqrt(out[i][0] * out[i][0] + out[i][1] * out[i][1]);
    }

    return fft_module;
}

std::vector<double> birkhoff_weights(unsigned int n_weights)
{
    // create a vector to store the birkhoff weights
    std::vector<double> weights(n_weights);
    for (unsigned int i = 0; i < n_weights; i++)
    {
        double t = (double)i / (double)(n_weights);
        weights[i] = exp(-1/(t * (1.0 - t)));
    }
    auto sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    // multiply by 1/sum to get the weights
    for (unsigned int i = 0; i < n_weights; i++)
    {
        weights[i] = weights[i] / sum;
    }
    return weights;
}

double birkhoff_tune(
    std::vector<double> const& x,
    std::vector<double> const& px
)
{
    // check if vectors are of the same size
    if (x.size() != px.size())
    {
        throw std::runtime_error("compute_tune: x and px must be of the same size");
    }

    // get size of vectors
    int n = x.size();

    // combine x and px into a complex vector
    std::vector<std::complex<double>> z(n);
    for (int i = 0; i < n; i++)
    {
        z[i] = std::complex<double>(x[i], px[i]);
    }

    // compute the angles of the complex numbers in the vector
    std::vector<double> theta(n);
    for (int i = 0; i < n; i++)
    {
        theta[i] = std::arg(z[i]);
    }

    // compute the vector of theta differences
    std::vector<double> dtheta(n - 1);
    for (int i = 0; i < n - 1; i++)
    {
        dtheta[i] = theta[i + 1] - theta[i];
        if(dtheta[i] < 0.0)
        {
            dtheta[i] += 2.0 * M_PI;
        }
    }

    // get the birkhoff weights
    std::vector<double> weights = birkhoff_weights(n - 1);

    // compute the sum of the weights times the theta differences
    double sum = 0.0;
    for (int i = 0; i < n - 1; i++)
    {
        sum += weights[i] * dtheta[i];
    }
    
    return 1 - sum / (2 * M_PI);
}


double A(const double &a, const double &b, const double &c)
{
    return (-(a + b * c) * (a - b) + b * sqrt(pow(c, 2) * pow((a + b), 2) - 2 * a * b * (2 * pow(c, 2) - c - 1))) / (pow(a, 2) + pow(b, 2) + 2 * a * b * c);
}

double interpolation(std::vector<double> const &data)
{
    // find index of maximum value in data
    unsigned int index = std::distance(data.begin(), std::max_element(data.begin(), data.end()));
    double N = double(data.size());

    // if the index is 0 or data.size() - 1, return
    if (index == 0)
    {
        return 1.0;
    }
    else if (index == data.size() - 1)
    {
        return 0.0;
    }

    // if the index is in the middle, interpolate
    unsigned int i1, i2;
    if (data[index - 1] > data[index + 1])
    {
        i1 = index - 1;
        i2 = index;
        index = index - 1;
    }
    else
    {
        i1 = index;
        i2 = index + 1;
    }

    double value = (
        (double(index) / N) + (1.0 / (2.0*M_PI)) * asin(
            A(data[i1], data[i2], cos(2.0*M_PI/N)) * sin(2.0*M_PI/N)
        )
    );

    return std::abs(1.0 - value);
}

double fft_tune(
    std::vector<double> const &x,
    std::vector<double> const &px,
    fftw_complex *in, fftw_complex *out, fftw_plan plan)
{
    // check if vectors are of the same size
    if (x.size() != px.size())
    {
        throw std::runtime_error("compute_tune: x and px must be of the same size");
    }
    // get size of vectors
    int n = x.size();

    // compute the magnitude of the fft
    std::vector<double> fft_mag = compute_fft(x, px, true, in, out, plan);

    // compute the interpolation
    return interpolation(fft_mag);
}

std::vector<double> birkhoff_tune_vec(std::vector<double> const &x, std::vector<double> const &px, std::vector<double> const &y, std::vector<double> const &py, std::vector<unsigned int> const &from, std::vector<unsigned int> const &to)
{
    // check if vectors are of the same size
    if (x.size() != px.size() || x.size() != y.size() || x.size() != py.size())
    {
        throw std::runtime_error("compute_tune: x, px, y, and py must be of the same size");
    }
    // if from and to are not empty, check if they are of the same size
    if (from.size() != to.size())
    {
        throw std::runtime_error("compute_tune: from and to must be of the same size");
    }

    // for each element in from and to
    std::vector<double> tunes;
    for (unsigned int i = 0; i < from.size(); i++)
    {
        // get the indices of the elements to be combined
        unsigned int from_index = from[i];
        unsigned int to_index = to[i];

        // check if from and to are valid
        if (from_index >= x.size() || to_index >= x.size())
        {
            throw std::runtime_error("compute_tune: from and to must be valid indices");
        }

        // get the vectors of the elements to be combined
        std::vector<double> x_sub = std::vector<double>(x.begin() + from_index, x.begin() + to_index + 1);
        std::vector<double> px_sub = std::vector<double>(px.begin() + from_index, px.begin() + to_index + 1);
        std::vector<double> y_sub = std::vector<double>(y.begin() + from_index, y.begin() + to_index + 1);
        std::vector<double> py_sub = std::vector<double>(py.begin() + from_index, py.begin() + to_index + 1);

        // check if there is nan in the vectors
        if (std::any_of(x_sub.begin(), x_sub.end(), [](double x) { return std::isnan(x); }) ||
            std::any_of(px_sub.begin(), px_sub.end(), [](double x) { return std::isnan(x); }) ||
            std::any_of(y_sub.begin(), y_sub.end(), [](double x) { return std::isnan(x); }) ||
            std::any_of(py_sub.begin(), py_sub.end(), [](double x) { return std::isnan(x); }))
        {
            tunes.push_back(std::numeric_limits<double>::quiet_NaN());
            tunes.push_back(std::numeric_limits<double>::quiet_NaN());
        }
        else
        {
            // compute the tune
            tunes.push_back(birkhoff_tune(x_sub, px_sub));
            tunes.push_back(birkhoff_tune(y_sub, py_sub));
        }
    }
    
    // compute the full tunes
    tunes.push_back(birkhoff_tune(x, px));
    tunes.push_back(birkhoff_tune(y, py));

    return tunes;
}

std::vector<double> fft_tune_vec(std::vector<double> const &x, std::vector<double> const &px, std::vector<double> const &y, std::vector<double> const &py, std::vector<unsigned int> const &from, std::vector<unsigned int> const &to,
std::map<unsigned int, std::tuple<fftw_complex *, fftw_complex *, fftw_plan>> plans)
{
    // check if vectors are of the same size
    if (x.size() != px.size() || x.size() != y.size() || x.size() != py.size())
    {
        throw std::runtime_error("compute_tune: x, px, y, and py must be of the same size");
    }
    // if from and to are not empty, check if they are of the same size
    if (from.size() != to.size())
    {
        throw std::runtime_error("compute_tune: from and to must be of the same size");
    }

    // for each element in from and to
    std::vector<double> tunes;
    for (unsigned int i = 0; i < from.size(); i++)
    {
        // get the indices of the elements to be combined
        unsigned int from_index = from[i];
        unsigned int to_index = to[i];
        unsigned int n = to_index - from_index + 1;
        // check if from and to are valid
        if (from_index >= x.size() || to_index >= x.size())
        {
            throw std::runtime_error("compute_tune: from and to must be valid indices");
        }

        // if any number in the sub vecros is a NaN, push back a quiet NaN
        if (std::any_of(x.begin() + from_index, x.end() + to_index + 1, [](double x){return std::isnan(x);}) ||
            std::any_of(px.begin() + from_index, px.end() + to_index + 1, [](double x){return std::isnan(x);}) ||
            std::any_of(y.begin() + from_index, y.end() + to_index + 1, [](double x){return std::isnan(x);}) ||
            std::any_of(py.begin() + from_index, py.end() + to_index + 1, [](double x){return std::isnan(x);}))
        {
            tunes.push_back(std::numeric_limits<double>::quiet_NaN());
            tunes.push_back(std::numeric_limits<double>::quiet_NaN());
            continue;
        }

        // get the vectors of the elements to be combined
        std::vector<double> x_sub = std::vector<double>(x.begin() + from_index, x.begin() + to_index + 1);
        std::vector<double> px_sub = std::vector<double>(px.begin() + from_index, px.begin() + to_index + 1);
        std::vector<double> y_sub = std::vector<double>(y.begin() + from_index, y.begin() + to_index + 1);
        std::vector<double> py_sub = std::vector<double>(py.begin() + from_index, py.begin() + to_index + 1);

        // compute the tune
        tunes.push_back(fft_tune(x_sub, px_sub, std::get<0>(plans[n]), std::get<1>(plans[n]), std::get<2>(plans[n])));
        tunes.push_back(fft_tune(y_sub, py_sub, std::get<0>(plans[n]), std::get<1>(plans[n]), std::get<2>(plans[n])));
    }

    if (std::any_of(x.begin(), x.end(), [](double x)
                    { return std::isnan(x); }) ||
        std::any_of(px.begin(), px.end(), [](double x)
                    { return std::isnan(x); }) ||
        std::any_of(y.begin(), y.end(), [](double x)
                    { return std::isnan(x); }) ||
        std::any_of(py.begin(), py.end(), [](double x)
                    { return std::isnan(x); }))
    {
        tunes.push_back(std::numeric_limits<double>::quiet_NaN());
        tunes.push_back(std::numeric_limits<double>::quiet_NaN());
    }
    else
    {
        // compute the full tunes
        // get the vectors of the elements to be combined
        std::vector<double> x_sub = std::vector<double>(x.begin() + 1, x.end());
        std::vector<double> px_sub = std::vector<double>(px.begin() + 1, px.end());
        std::vector<double> y_sub = std::vector<double>(y.begin() + 1, y.end());
        std::vector<double> py_sub = std::vector<double>(py.begin() + 1, py.end());

        tunes.push_back(fft_tune(x_sub, px_sub, std::get<0>(plans[x.size() - 1]), std::get<1>(plans[x.size() - 1]), std::get<2>(plans[x.size() - 1])));
        tunes.push_back(fft_tune(y_sub, py_sub, std::get<0>(plans[x.size() - 1]), std::get<1>(plans[x.size() - 1]), std::get<2>(plans[x.size() - 1])));
    }
    return tunes;
}


std::array<double, 2> get_tunes(std::vector<double> x, std::vector<double> px)
{
    // check if vectors are of the same size
    if (x.size() != px.size())
    {
        throw std::runtime_error("get_tunes: x and px must be of the same size");
    }
    auto size = x.size();
    // if any number in the sub vecros is a NaN, push back a quiet NaN
    if (std::any_of(x.begin(), x.end(), [](double x){return std::isnan(x);}) ||
        std::any_of(px.begin(), px.end(), [](double x){return std::isnan(x);}))
    {
        return std::array<double, 2>{std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
    }

    double birkhoff_tune_val = birkhoff_tune(x, px);

    fftw_complex *in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * size);
    fftw_complex *out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * size);
    fftw_plan plan = fftw_plan_dft_1d(size, in, out, FFTW_FORWARD, FFTW_MEASURE);

    double fft_tune_val = fft_tune(x, px, in, out, plan);

    // free the memory
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    return std::array<double, 2>{fft_tune_val, birkhoff_tune_val};
}