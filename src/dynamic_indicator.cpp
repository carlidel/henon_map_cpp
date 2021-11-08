#include "dynamic_indicator.h"

// mutex lock for fft
std::mutex fft_lock;


std::array<std::vector<double>, 2> compute_fft(std::vector<double> const &real_signal, std::vector<double> const &imag_signal, bool hanning_window) {
    // check if the signal is of the correct length
    if (real_signal.size() != imag_signal.size()) {
        throw std::invalid_argument("real and imag signals must have the same length");
    }
    int N = real_signal.size();
    
    std::vector<double> fft_real(N);
    std::vector<double> fft_imag(N);

    // apparently, we need a mutex here...
    fft_lock.lock();

    fftw_complex *in, *out;
    fftw_plan p;
    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    // free the mutex
    fft_lock.unlock();

    for (int i = 0; i < N; i++) {
        in[i][0] = real_signal[i];
        in[i][1] = imag_signal[i];
    }

    // if hanning window is true, apply hanning window
    if (hanning_window) {
        for (int i = 0; i < N; i++) {
            in[i][0] *= 0.5 * (1 - cos(2 * M_PI * i / (N - 1)));
            in[i][1] *= 0.5 * (1 - cos(2 * M_PI * i / (N - 1)));
        }
    }

    fftw_execute(p);
    for (int i = 0; i < N; i++) {
        fft_real[i] = out[i][0];
        fft_imag[i] = out[i][1];
    }
    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);

    return {fft_real, fft_imag};
}

std::vector<double> birkhoff_weights(unsigned int n_weights)
{
    // create a vector to store the birkhoff weights
    std::vector<double> weights(n_weights);
    for (unsigned int i = 0; i < n_weights; i++)
    {
        if (i == 0)
        {
            weights[i] = 0.0;
        }
        else if (i == n_weights - 1)
        {
            weights[i] = 0.0;
        }
        else
        {
            // TODO: check if this is correct
            double t = (double)i / (double)(n_weights - 1);
            weights[i] = exp((t * (1.0 - t)));
        }
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


double interpolation(std::vector<double> const &data, int index)
{
    // check if the index is valid
    if (index < 0 || index >= data.size())
    {
        throw std::invalid_argument("interpolation: index must be between 0 and data.size()");
    }

    // if the index is 0 or data.size() - 1, return the value at that index
    if (index == 0)
    {
        return data[0];
    }
    else if (index == data.size() - 1)
    {
        return data[data.size() - 1];
    }

    // if the index is in the middle, interpolate
    double cf1 = data[index - 1];
    double cf2 = data[index];
    double cf3 = data[index + 1];

    double p1, p2, nn;
    if (cf3 > cf1)
    {
        p1 = cf2;
        p2 = cf3;
        nn = index;
    }
    else
    {
        p1 = cf1;
        p2 = cf2;
        nn = index - 1;
    }
    double p3 = cos(2 * M_PI / data.size());
    
    // TODO: check if this is correct
    double value = ((nn / data.size()) + (1.0 / M_1_PI) * asin(sin(2 * M_PI / data.size()) * ((-(p1 + p2 * p3) * (p1 - p2) + p2 * sqrt(p3 * p3 * p3 * (p1 + p2) * (p1 + p2) - 2 * p1 * p2 * (2 * p3 * p3 - p3 - 1))) / (p1 * p1 + p2 * p2 + 2 * p1 * p2 * p3))));

    return std::abs(1.0 - value);
}


double fft_tune(
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

    // get fft of the vectors
    std::array<std::vector<double>, 2> fft_x = compute_fft(x, px, true);

    // compute the magnitude of the fft
    std::vector<double> fft_mag(n);
    for (int i = 0; i < n; i++)
    {
        fft_mag[i] = std::sqrt(
            fft_x[0][i] * fft_x[0][i] + fft_x[1][i] * fft_x[1][i]
        );
    }

    // find the index of the maximum magnitude
    int max_index = 0;
    for (int i = 1; i < n; i++)
    {
        if (fft_mag[i] > fft_mag[max_index])
        {
            max_index = i;
        }
    }

    // compute the interpolation
    return interpolation(fft_mag, max_index);
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

        // compute the tune
        tunes.push_back(birkhoff_tune(x_sub, px_sub));
        tunes.push_back(birkhoff_tune(y_sub, py_sub));
    }
    
    // compute the full tunes
    tunes.push_back(birkhoff_tune(x, px));
    tunes.push_back(birkhoff_tune(y, py));

    return tunes;
}

std::vector<double> fft_tune_vec(std::vector<double> const &x, std::vector<double> const &px, std::vector<double> const &y, std::vector<double> const &py, std::vector<unsigned int> const &from, std::vector<unsigned int> const &to)
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

        // compute the tune
        tunes.push_back(fft_tune(x_sub, px_sub));
        tunes.push_back(fft_tune(y_sub, py_sub));
    }

    // compute the full tunes
    tunes.push_back(fft_tune(x, px));
    tunes.push_back(fft_tune(y, py));

    return tunes;
}