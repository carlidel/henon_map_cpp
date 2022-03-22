#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/stl_bind.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

#include "henon.h"
#include "modulation.h"
#include "dynamic_indicator.h"

bool has_cuda_error_happened()
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return true;
    }
    return false;
}

bool is_cuda_device_available()
{
    int count;
    cudaGetDeviceCount(&count);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        return false;
    }
    std::cout << "CUDA devices available: " << count << std::endl;
    return count > 0;
}

namespace py = pybind11;

struct py_particles_4d : public particles_4d
{
public:
    /* Inherit the constructors */
    using particles_4d::particles_4d;

    void reset() override { PYBIND11_OVERRIDE(void, particles_4d, reset, ); }
    void add_ghost(const double &displacement_module, const std::string &direction) override
    {
        PYBIND11_OVERRIDE(void, particles_4d, add_ghost, displacement_module, direction);
    }

    void renormalize(const double &module_target) override
    {
        PYBIND11_OVERRIDE(void, particles_4d, renormalize, module_target);
    }

    const std::vector<std::vector<double>> get_displacement_module() const override
    {
        PYBIND11_OVERRIDE(const std::vector<std::vector<double>>, particles_4d, get_displacement_module, );
    }
    const std::vector<std::vector<std::vector<double>>> get_displacement_direction() const override
    {
        PYBIND11_OVERRIDE(const std::vector<std::vector<std::vector<double>>>, particles_4d, get_displacement_direction);
    }

    const std::vector<double> get_x() const override
    {
        PYBIND11_OVERRIDE(const std::vector<double>, particles_4d, get_x);
    }
    const std::vector<double> get_px() const override
    {
        PYBIND11_OVERRIDE(const std::vector<double>, particles_4d, get_px);
    }
    const std::vector<double> get_y() const override
    {
        PYBIND11_OVERRIDE(const std::vector<double>, particles_4d, get_y);
    }
    const std::vector<double> get_py() const override
    {
        PYBIND11_OVERRIDE(const std::vector<double>, particles_4d, get_py);
    }
    const std::vector<unsigned int> get_steps() const override
    {
        PYBIND11_OVERRIDE(const std::vector<unsigned int>, particles_4d, get_steps);
    }

    const std::vector<uint8_t> get_valid() const override
    {
        PYBIND11_OVERRIDE(const std::vector<uint8_t>, particles_4d, get_valid);
    }
    const std::vector<uint8_t> get_ghost() const override
    {
        PYBIND11_OVERRIDE(const std::vector<uint8_t>, particles_4d, get_ghost);
    }

    const std::vector<size_t> get_idx() const override
    {
        PYBIND11_OVERRIDE(const std::vector<size_t>, particles_4d, get_idx);
    }
    const std::vector<size_t> get_idx_base() const override
    {
        PYBIND11_OVERRIDE(const std::vector<size_t>, particles_4d, get_idx_base);
    }

    const size_t &get_n_particles() const override
    {
        PYBIND11_OVERRIDE(const size_t &, particles_4d, get_n_particles);
    }
    const size_t &get_n_ghosts_per_particle() const override
    {
        PYBIND11_OVERRIDE(const size_t &, particles_4d, get_n_ghosts_per_particle);
    }
};

struct py_henon_tracker : public henon_tracker
{
    using henon_tracker::henon_tracker;

    void compute_a_modulation(unsigned int N, double omega_x, double omega_y, std::string modulation_kind = "sps", double omega_0 = NAN, double epsilon = 0.0, unsigned int offset = 0) override
    {
        PYBIND11_OVERRIDE(
            void,
            henon_tracker,
            compute_a_modulation,
            N, omega_x, omega_y, modulation_kind, omega_0, epsilon, offset);
    }
};

struct py_henon_tracker_gpu : public henon_tracker_gpu
{
    using henon_tracker_gpu::henon_tracker_gpu;

    void compute_a_modulation(unsigned int N, double omega_x, double omega_y, std::string modulation_kind = "sps", double omega_0 = NAN, double epsilon = 0.0, unsigned int offset = 0) override
    {
        PYBIND11_OVERRIDE(
            void,
            henon_tracker_gpu,
            compute_a_modulation,
            N, omega_x, omega_y, modulation_kind, omega_0, epsilon, offset);
    }
};

PYBIND11_MODULE(henon_map_engine, m)
{
    m.doc() = "Henon map engine";
    m.def("is_cuda_device_available", &is_cuda_device_available);

    m.def("basic_modulation", &basic_modulation, "Basic modulation",
          py::arg("tune"), py::arg("omega"), py::arg("epsilon"), py::arg("start"),
          py::arg("end"));
    m.def("sps_modulation", &sps_modulation, "SPS modulation",
          py::arg("tune"), py::arg("epsilon"), py::arg("start"), py::arg("end"));
    m.def("gaussian_modulation", &gaussian_modulation, "Gaussian modulation",
          py::arg("tune"), py::arg("sigma"), py::arg("start"), py::arg("end"));
    m.def("uniform_modulation", &uniform_modulation, "Uniform modulation",
          py::arg("from"), py::arg("to"), py::arg("start"), py::arg("end"));

    m.def("birkhoff_weights", &birkhoff_weights, "Birkhoff weights", py::arg("n_weights"));

    m.def("get_tunes", &get_tunes, "Get tunes",
          py::arg("x"), py::arg("px"));

    py::class_<particles_4d, py_particles_4d>(m, "particles_4d")
        .def(py::init<
             const std::vector<double> &,
             const std::vector<double> &,
             const std::vector<double> &,
             const std::vector<double> &>())
        .def("reset", &particles_4d::reset)
        .def("add_ghost", &particles_4d::add_ghost,
             py::arg("displacement_module"), py::arg("direction"))
        .def("renormalize", &particles_4d::renormalize, py::arg("module_target"))
        .def("get_displacement_module", &particles_4d::get_displacement_module)
        .def("get_displacement_direction", &particles_4d::get_displacement_direction)
        .def("get_x", &particles_4d::get_x)
        .def("get_px", &particles_4d::get_px)
        .def("get_y", &particles_4d::get_y)
        .def("get_py", &particles_4d::get_py)
        .def("get_steps", &particles_4d::get_steps)
        .def("get_valid", &particles_4d::get_valid)
        .def("get_ghost", &particles_4d::get_ghost)
        .def("get_idx", &particles_4d::get_idx)
        .def("get_idx_base", &particles_4d::get_idx_base)
        .def("get_n_particles", &particles_4d::get_n_particles)
        .def("get_n_ghosts_per_particle", &particles_4d::get_n_ghosts_per_particle);

    py::class_<matrix_4d_vector>(m, "matrix_4d_vector")
        .def(py::init<size_t>())
        .def("reset", &matrix_4d_vector::reset)
        .def("multiply", &matrix_4d_vector::multiply, py::arg("matrix"))
        .def("structured_multiply",
             static_cast<void (matrix_4d_vector::*)(const henon_tracker &, const particles_4d &, const double &)>(&matrix_4d_vector::structured_multiply),
             py::arg("tracker"), py::arg("particles"), py::arg("mu"))
        .def("structured_multiply",
             static_cast<void (matrix_4d_vector::*)(const henon_tracker &, const particles_4d &, const double &)>(&matrix_4d_vector::structured_multiply),
             py::arg("tracker"), py::arg("particles"), py::arg("mu"))
        .def("get_matrix", &matrix_4d_vector::get_matrix)
        .def("get_vector", &matrix_4d_vector::get_vector, py::arg("vector"));

    py::class_<matrix_4d_vector_gpu>(m, "matrix_4d_vector_gpu")
        .def(py::init<size_t>())
        .def("reset", &matrix_4d_vector_gpu::reset)
        .def("structured_multiply", &matrix_4d_vector_gpu::structured_multiply, py::arg("tracker"), py::arg("particles"), py::arg("mu"))
        .def("get_matrix", &matrix_4d_vector_gpu::get_matrix)
        .def("get_vector", &matrix_4d_vector_gpu::get_vector, py::arg("vector"));

    py::class_<particles_4d_gpu, particles_4d>(m, "particles_4d_gpu")
        .def(py::init<
             const std::vector<double> &,
             const std::vector<double> &,
             const std::vector<double> &,
             const std::vector<double> &>());

    py::class_<henon_tracker, py_henon_tracker>(m, "henon_tracker")
        .def(py::init<
             unsigned int,
             double,
             double,
             std::string,
             double,
             double,
             unsigned int>())
        .def("compute_a_modulation", &henon_tracker::compute_a_modulation,
             "Compute a modulation",
             py::arg("N"), py::arg("omega_x"), py::arg("omega_y"), py::arg("modulation_kind"), py::arg("omega_0"), py::arg("epsilon"), py::arg("offset"))
        .def("track", &henon_tracker::track, "Track particles", py::arg("particles"), py::arg("n_turns"), py::arg("mu"), py::arg("barrier") = 100.0, py::arg("kick_module") = NAN, py::arg("inverse") = false)
        .def("birkhoff_tunes", &henon_tracker::birkhoff_tunes, "Compute birkhoff tunes", py::arg("particles"), py::arg("n_turns"), py::arg("mu"), py::arg("barrier") = 100.0, py::arg("kick_module") = NAN, py::arg("inverse") = false, py::arg("from_idx") = std::vector<unsigned int>(), py::arg("to_idx") = std::vector<unsigned int>())
        .def("get_tangent_matrix", &henon_tracker::get_tangent_matrix, "get tangent matrix", py::arg("particles"), py::arg("mu"));

    py::class_<henon_tracker_gpu, py_henon_tracker_gpu>(m, "henon_tracker_gpu")
        .def(py::init<
             unsigned int,
             double,
             double,
             std::string,
             double,
             double,
             unsigned int>())
        .def("compute_a_modulation", &henon_tracker_gpu::compute_a_modulation,
             "Compute a modulation",
             py::arg("N"), py::arg("omega_x"), py::arg("omega_y"), py::arg("modulation_kind"), py::arg("omega_0"), py::arg("epsilon"), py::arg("offset"))
        .def("get_tangent_matrix", &henon_tracker_gpu::get_tangent_matrix, "Get tangent matrix", py::arg("particles"), py::arg("mu"));

    py::class_<storage_4d>(m, "storage_4d")
        .def(py::init<size_t>())
        .def("store",
            static_cast<void (storage_4d::*)(const particles_4d &)>(&storage_4d::store),
            "Store particles")
        .def("store",
            static_cast<void (storage_4d::*)(const particles_4d_gpu &)>(& storage_4d::store),
            "Store particles")
        .def("tune_fft", &storage_4d::tune_fft, "Tune FFT", py::arg("from_idx"), py::arg("to_idx"))
        .def("tune_birkhoff", &storage_4d::tune_birkhoff, "Tune birkhoff", py::arg("from_idx"), py::arg("to_idx"))
        .def("get_x", &storage_4d::get_x)
        .def("get_px", &storage_4d::get_px)
        .def("get_y", &storage_4d::get_y)
        .def("get_py", &storage_4d::get_py);
}